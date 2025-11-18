import logging

from sqlalchemy import create_engine, DateTime
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from app.schemas.meta import Base, DataInfo
from app.config import settings

connect_args = {"check_same_thread": False}
sqlite_url: str = f"sqlite:///{settings.SQLITE_FILE_NAME}"
engine = create_engine(sqlite_url, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush= False, autocommit=False)


def create_db_and_tables():
    """Create db and tables."""
    try:
        Base.metadata.create_all(engine)

    except SQLAlchemyError as e:
        logging.error(f"Database initialization failed while creating tables: {e}")
        raise
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")
        raise


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_metadata_to_db(origin_text: str,topics: list[str], tags: list[str], description: str, extract_time: DateTime):
    import json
    generator = get_db()
    Session = next(generator)
    # JSON 형으로 바꾸기.. convert 문제로 일단은 text형으로 정의
    topics_json = json.dumps(topics, ensure_ascii=False) if topics else "[]"
    tags_json = json.dumps(tags, ensure_ascii=False) if tags else "[]"
    
    db_data_info = DataInfo(origin_text=origin_text, topics=topics_json, tags=tags_json, description=description, extract_time=extract_time)
    Session.add(db_data_info)
    Session.commit()
    Session.refresh(db_data_info)
    return db_data_info
