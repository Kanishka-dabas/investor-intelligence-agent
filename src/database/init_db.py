from src.database.connection import engine
from src.database.models import Base
from loguru import logger


def init_db():
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created (or already exist).")


if __name__ == "__main__":
    init_db()