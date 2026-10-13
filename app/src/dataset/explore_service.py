"""데이터 찾기 — 분야 탐색, 검색, 상세, 함께 쓸 수 있는 데이터.

함께 쓸 데이터는 관계 유형별로 묶는다. 묶음 순서와 문구는 관계 유형 정의서의 화면 배치
(속성 보완 → 집계 맥락 → 인접 맥락 → 동종 판본)를 따른다.
"""

import csv
import io
from collections import Counter
from typing import Any

from app.exceptions.service import DatasetNotFoundError
from app.repositories.dataset_repository import DatasetRepository, get_dataset_repository
from app.repositories.relation_repository import RelationRepository, get_relation_repository
from app.schemas.dataset import DatasetProfile, Level1RelationType, column_jaccard
from app.schemas.explore import (
    DatasetCard,
    DatasetDetail,
    LinkableGroup,
    LinkableItem,
    LinkableResult,
    SearchPage,
    TaxonomyNode,
)
from app.src.dataset.domain_classifier import SEPARATOR, TAXONOMY
from app.src.embedding.embedding_service import embedding_service

# (유형, 묶음 이름, 사용자 안내) — 화면 배치 순서
_GROUPS = (
    (Level1RelationType.DETAIL_FOR, "속성 보완", "같은 대상의 다른 정보"),
    (Level1RelationType.STATISTIC_FOR, "집계 맥락", "비교 기준이 되는 통계"),
    (Level1RelationType.ADJACENT_TO, "인접 맥락", "이어서 보면 좋은 관련 데이터"),
    (Level1RelationType.SAME_KIND, "동종 판본", "같은 종류의 다른 지역·기관 데이터"),
)

_CANDIDATE_COUNT = 10
_POOL_SIZE = 20

_CSV_HEADER = ["기준 데이터", "관계 유형", "관계 묶음", "하위라벨", "연계 데이터", "제공기관", "신뢰도", "결합 키", "활용 설명", "원본 URL"]


def to_card(profile: DatasetProfile, relation_count: int = 0) -> DatasetCard:
    """프로파일 → 카드."""
    return DatasetCard(
        **profile.model_dump(include=set(DatasetCard.model_fields) - {"relation_count", "similarity"}),
        relation_count=relation_count,
    )


def _join_keys(evidence: dict[str, Any], anchor_is_a: bool) -> list[str]:
    """근거의 컬럼 매칭을 '기준 컬럼 = 상대 컬럼' 으로. 매칭이 없으면 LLM 인용 컬럼."""
    matches = [
        m for m in (evidence.get("matched_columns") or []) + (evidence.get("ai_matched_columns") or [])
        if isinstance(m, dict) and m.get("column_a") and m.get("column_b")
    ]
    keys = []
    for m in matches:
        mine, theirs = (m["column_a"], m["column_b"]) if anchor_is_a else (m["column_b"], m["column_a"])
        keys.append(mine if mine == theirs else f"{mine} = {theirs}")
    return list(dict.fromkeys(keys)) or [str(c) for c in evidence.get("evidence_columns") or []]


class ExploreService:
    """데이터 찾기 화면의 조회 로직."""

    def __init__(
        self,
        dataset_repo: DatasetRepository | None = None,
        relation_repo: RelationRepository | None = None,
    ):
        """데이터셋·관계 저장소를 주입받아 초기화."""
        self.dataset_repo = dataset_repo or get_dataset_repository()
        self.relation_repo = relation_repo or get_relation_repository()

    def _profile(self, table_name: str) -> DatasetProfile:
        """원장 프로파일 조회 (임베딩 포함).

        Raises:
            DatasetNotFoundError: 원장에 없는 데이터셋
        """
        profile = self.dataset_repo.get(table_name)
        if not profile:
            raise DatasetNotFoundError(table_name)
        return profile

    def relation_counts(self, table_names: list[str]) -> Counter:
        """데이터셋별 확정 관계 수."""
        names = set(table_names)
        return Counter(
            t for e in self.relation_repo.edges(table_names) for t in (e["source"], e["target"]) if t in names
        )

    def overview(self) -> dict[str, Any]:
        """홈·검색 화면 공통 — 전체 수, 분류표(분야별 수), 제공기관 목록."""
        counts = self.dataset_repo.category_counts()
        profiles = self.dataset_repo.list_all()
        return {
            "dataset_count": len(profiles),
            "relation_count": len(self.relation_repo.edges()),
            "taxonomy": [
                TaxonomyNode(
                    name=major,
                    count=counts.get(major, 0),
                    children=[TaxonomyNode(name=minor, count=counts.get(f"{major}{SEPARATOR}{minor}", 0))
                              for minor in minors],
                ).model_dump()
                for major, minors in TAXONOMY.items()
            ],
            "providers": sorted({p.provider for p in profiles if p.provider}),
        }

    def search(
        self,
        query: str = "",
        category: str = "",
        subcategory: str = "",
        party: str = "",
        provider: str = "",
        limit: int = 20,
        offset: int = 0,
    ) -> SearchPage:
        """분야·조건 필터 + 검색어 결과 한 페이지."""
        query = query.strip()
        total, rows = self.dataset_repo.search(
            query=query,
            query_embedding=embedding_service.generate_embedding(query) if query else None,
            category=category, subcategory=subcategory, party=party, provider=provider,
            limit=limit, offset=offset,
        )
        return SearchPage(
            total=total,
            items=[
                DatasetCard(
                    table_name=r["file_name"], display_name=r["display_name"] or "",
                    description=r["description"] or "", table_summary=r["table_summary"] or "",
                    provider=r["publisher"] or "", party=r["party"] or "", source=r["source"],
                    categories=r["theme"], record_count=r["record_count"] or 0,
                    column_count=r["column_count"] or 0, landing_page=r["landing_page"] or "",
                    keywords=r["keywords"], relation_count=r["relation_count"], similarity=r["similarity"],
                    created_at=r["created_at"].isoformat(),
                )
                for r in rows
            ],
        )

    def detail(self, table_name: str) -> DatasetDetail:
        """상세 — 카드 + 컬럼·샘플."""
        profile = self._profile(table_name)
        return DatasetDetail(
            **to_card(profile, self.relation_counts([table_name])[table_name]).model_dump(),
            column_names=profile.column_names,
            sample_records=profile.sample_records,
        )

    def linkable(self, table_name: str) -> LinkableResult:
        """기준 데이터와 함께 쓸 수 있는 데이터 — 확정 관계는 유형별로, 미판정 유사 후보는 따로."""
        anchor = self._profile(table_name)
        edges = self.relation_repo.edges([table_name])

        grouped: dict[str, list[LinkableItem]] = {t.value: [] for t, _, _ in _GROUPS}
        for e in edges:
            anchor_is_a = e["source"] == table_name
            other = self.dataset_repo.get(e["target"] if anchor_is_a else e["source"])
            head, _, sub = e["relation_type"].partition(":")
            if not other or head not in grouped:
                continue
            grouped[head].append(LinkableItem(
                table_name=other.table_name,
                display_name=other.display_name,
                provider=other.provider,
                description=other.table_summary or other.description,
                landing_page=other.landing_page,
                relation_type=e["relation_type"],
                sub_label=sub,
                score=e["score"],
                join_keys=_join_keys(e["evidence"], anchor_is_a),
                reason=str(e["evidence"].get("usage") or e["evidence"].get("reason") or ""),
            ))

        related = {e["source"] for e in edges} | {e["target"] for e in edges}
        knn = (
            self.dataset_repo.search_knn(
                anchor.table_embedding, top_k=_CANDIDATE_COUNT, party="3rd", exclude_table_name=table_name
            )
            if anchor.table_embedding
            else []
        )
        return LinkableResult(
            anchor=to_card(anchor, len(edges)),
            groups=[
                LinkableGroup(
                    relation_type=t.value, label=label, description=desc,
                    items=sorted(grouped[t.value], key=lambda i: i.score, reverse=True),
                )
                for t, label, desc in _GROUPS
            ],
            candidates=[
                LinkableItem(
                    table_name=r.table_name, display_name=r.display_name, provider=r.provider,
                    description=r.table_summary or r.description, landing_page=r.landing_page, score=r.score,
                )
                for r in knn if r.table_name not in related
            ],
            graph={
                "nodes": [{"id": n, "seed": n == table_name} for n in related | {table_name}],
                "edges": [{k: e[k] for k in ("source", "target", "relation_type", "score")} for e in edges],
            },
        )

    def generation_targets(self, table_name: str) -> list[str]:
        """관계 생성 대상 — 기준 + 의미가 가까운 외부 개방 데이터 상위 N건 + 컬럼이 겹치는 외부 개방 데이터.

        3rd 파티 전체와 쌍을 만들면 쌍마다 LLM 판정이 붙어 사용자가 기다릴 수 없다.
        공통 컬럼은 의미 유사도와 무관한 구조 신호(DETAIL_FOR 근거)라 순위 밖이어도 넣는다.
        """
        anchor = self._profile(table_name)
        semantic = (
            self.dataset_repo.search_knn(
                anchor.table_embedding, top_k=_POOL_SIZE, party="3rd", exclude_table_name=table_name
            )
            if anchor.table_embedding
            else []
        )
        structural = [
            p.table_name for p in self.dataset_repo.list_all()
            if p.party == "3rd" and p.table_name != table_name
            and column_jaccard(anchor.column_names, p.column_names) > 0
        ]
        return list(dict.fromkeys([table_name] + [r.table_name for r in semantic] + structural))

    @staticmethod
    def to_csv(result: LinkableResult) -> str:
        """판정된 연계 데이터를 CSV 로 (엑셀 호환 BOM 포함)."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(_CSV_HEADER)
        anchor = result.anchor.display_name or result.anchor.table_name
        writer.writerows(
            [
                anchor, group.relation_type, group.label, item.sub_label,
                item.display_name or item.table_name, item.provider, f"{item.score:.2f}",
                "; ".join(item.join_keys), item.reason, item.landing_page,
            ]
            for group in result.groups
            for item in group.items
        )
        return "﻿" + buf.getvalue()


explore_service = ExploreService()
