import os
import logging
import ssl
from typing import AsyncGenerator
from urllib.parse import urlparse, parse_qs

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Get database URL from environment variables
raw_db_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost/research_papers")

# Parse the URL to remove sslmode parameter if present
parsed_url = urlparse(raw_db_url)
query_params = parse_qs(parsed_url.query)

# Remove sslmode from query parameters
if 'sslmode' in query_params:
    ssl_mode = query_params.pop('sslmode')[0]
    logger.info(f"Removed sslmode={ssl_mode} from connection string, will handle SSL separately")

# Reconstruct URL without sslmode
from urllib.parse import urlencode, urlunparse
new_query = urlencode(query_params, doseq=True)
clean_url_parts = list(parsed_url)
clean_url_parts[4] = new_query  # index 4 is the query part
DATABASE_URL = urlunparse(clean_url_parts)

# Determine if SSL is needed (for cloud databases like NeonDB)
is_cloud_db = 'neon.tech' in parsed_url.netloc or 'amazonaws.com' in parsed_url.netloc
connect_args = {}

if is_cloud_db:
    # Create SSL context for secure connections
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    connect_args["ssl"] = ssl_context
    logger.info("SSL context created for cloud database connection")

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False,
    future=True,
)

# Create async session factory
async_session_factory = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)

# Create base class for declarative models
Base = declarative_base()

async def init_db():
    """Initialize the database by creating all tables."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()