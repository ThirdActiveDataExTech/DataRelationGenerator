"""Level 1 ↔ Level 2 게이팅 서비스.

설계 철학(§2): "먼저 Level 1로 데이터셋 간 연결 구조가 만들어지면, 그 위에서 Level 2가
레코드 수준으로 관계를 정밀화한다." 따라서 Level 2(레코드 간) 후보 발굴은 기준 레코드가
속한 데이터셋과 Level 1으로 연결된 데이터셋의 레코드로만 한정한다.

이 모듈은 원장의 확정 데이터셋 관계(dcat_dataset_relation)로 특정 데이터셋과 직접
연결된 데이터셋 집합을 계산한다.
"""

from app.repositories.relation_repository import RelationRepository, get_relation_repository


class Level1GatingService:
    """Level 1 데이터셋 연결 구조를 기반으로 Level 2 검색 범위를 산출."""

    def __init__(self, relation_repo: RelationRepository | None = None) -> None:
        """게이팅 서비스 초기화."""
        self.relation_repo = relation_repo or get_relation_repository()

    def get_connected_tables(self, table_name: str) -> set[str]:
        """주어진 데이터셋과 Level 1으로 연결된 데이터셋명 집합을 반환 (양방향).

        Args:
            table_name: 기준 데이터셋(테이블)명

        Returns:
            연결된 데이터셋명 집합 (기준 데이터셋 자신은 제외)
        """
        edges = self.relation_repo.edges([table_name])
        return {e["target"] if e["source"] == table_name else e["source"] for e in edges} - {table_name}

    def compute_allowed_tables(self, source_table: str) -> set[str]:
        """Level 2 후보가 속할 수 있는 데이터셋 집합을 산출.

        기준 데이터셋과 Level 1으로 연결된 데이터셋들 + 기준 데이터셋 자신(동일 데이터셋
        내 중복/참조 레코드 발견을 위해 포함).

        Args:
            source_table: 기준 레코드가 속한 데이터셋명

        Returns:
            허용 데이터셋명 집합
        """
        allowed = self.get_connected_tables(source_table)
        allowed.add(source_table)
        return allowed
