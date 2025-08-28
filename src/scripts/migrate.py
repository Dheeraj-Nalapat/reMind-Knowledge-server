import psycopg2

from src.common.logger.logger import get_logger
from src.config import Config

logger = get_logger(__name__)

MIGRATION_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS notion_pages (
    page_id UUID PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding VECTOR(1536),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);
"""


def run_migration():
    logger.info("Connecting to database...")
    conn = psycopg2.connect(Config.PG_URL)
    cur = conn.cursor()
    logger.info("Running migration...")
    cur.execute(MIGRATION_SQL)
    conn.commit()
    cur.close()
    conn.close()
    logger.info("Migration completed successfully.")


if __name__ == "__main__":
    run_migration()
