from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import settings

engine = create_engine(settings.database.url.get_secret_value(), pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)