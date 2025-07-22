import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor
from src.config import Config
from src.common.logger.logger import get_logger

logger = get_logger(__name__)

class DatabaseConnection:
    _pool = None
    
    @classmethod
    def get_pool(cls):
        """Get or create a connection pool"""
        if cls._pool is None:
            try:
                cls._pool = SimpleConnectionPool(
                    minconn=1,
                    maxconn=10,
                    dsn=Config.PG_URL,
                    cursor_factory=RealDictCursor
                )
                logger.info("Database connection pool created successfully")
            except Exception as e:
                logger.error(f"Failed to create database connection pool: {e}")
                raise
        return cls._pool
    
    @classmethod
    def get_connection(cls):
        """Get a connection from the pool"""
        pool = cls.get_pool()
        return pool.getconn()
    
    @classmethod
    def return_connection(cls, conn):
        """Return a connection to the pool"""
        if cls._pool:
            cls._pool.putconn(conn)
    
    @classmethod
    def close_pool(cls):
        """Close the connection pool"""
        if cls._pool:
            cls._pool.closeall()
            cls._pool = None
            logger.info("Database connection pool closed")

def get_db_connection():
    """Helper function to get a database connection"""
    return DatabaseConnection.get_connection()

def return_db_connection(conn):
    """Helper function to return a database connection"""
    DatabaseConnection.return_connection(conn) 