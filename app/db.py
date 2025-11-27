from sqlalchemy import create_engine

from sqlalchemy.orm import Session, sessionmaker

from app.config import settings

connect_args = {"check_same_thread": False}
sqlite_url: str = f"sqlite:///{settings.SQLITE_FILE_NAME}"
engine = create_engine(sqlite_url, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
