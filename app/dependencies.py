from typing import Annotated, Generator, Optional
from functools import lru_cache

from fastapi import Security, Depends
from fastapi.security import APIKeyHeader
from sqlmodel import Session
from langfuse import Langfuse
from app.db import engine
from app.utils.authentication import token_validation
from openai import OpenAI
from app.clients import langfuse, openai

header_scheme = APIKeyHeader(name="x-token")


async def get_token_header(x_token: Annotated[str, Security(header_scheme)]):
    await token_validation(x_token)


def get_session() -> Generator[Session, None, None]:
    """Get session"""
    with Session(engine) as session:
        yield session


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    return openai.get_client()


# Client Instances
openai_client: OpenAI = get_openai_client()
langfuse_client: Optional[Langfuse] = langfuse.get_client()

SessionDep = Annotated[Session, Depends(get_session)]
