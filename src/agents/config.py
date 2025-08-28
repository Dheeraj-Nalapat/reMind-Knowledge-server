"""
Configuration file for the ReMind Knowledge Processor agents.
"""

import os
from typing import Dict, Any

# Agent Configuration
AGENT_CONFIG = {
    "data_structurer": {
        "model": "gemini-2.0-flash",
        "max_content_length": 10000,
        "enable_entity_extraction": True,
        "enable_summarization": True,
        "quality_threshold": 0.7,
    },
    "notion_builder": {
        "model": "gemini-2.0-flash",
        "default_parent_id": os.getenv("NOTION_DEFAULT_PARENT_ID"),
        "enable_rich_formatting": True,
        "auto_categorize": True,
        "max_blocks_per_page": 100,
    },
    "knowledge_processor": {
        "model": "gemini-2.0-flash",
        "embedding_model": "text-embedding-ada-002",
        "enable_semantic_search": True,
        "cache_embeddings": True,
        "batch_size": 10,
    },
    "coordinator": {
        "model": "gemini-2.0-flash",
        "enable_parallel_processing": True,
        "max_retries": 3,
        "timeout_seconds": 300,
        "enable_monitoring": True,
    },
}

# Workflow Configuration
WORKFLOW_CONFIG = {
    "steps": [
        "data_structuring",
        "notion_creation",
        "knowledge_processing",
        "finalization",
    ],
    "error_handling": {
        "retry_failed_steps": True,
        "max_retries": 3,
        "continue_on_error": False,
    },
    "performance": {
        "enable_caching": True,
        "parallel_processing": True,
        "batch_processing": True,
    },
}

# Content Processing Configuration
CONTENT_CONFIG = {
    "supported_formats": ["text", "markdown", "html"],
    "max_file_size": 1024 * 1024,  # 1MB
    "content_validation": {
        "min_length": 10,
        "max_length": 50000,
        "require_title": False,
    },
    "categorization": {
        "auto_categorize": True,
        "default_category": "general",
        "categories": [
            "technology",
            "business",
            "education",
            "personal",
            "research",
            "creative",
            "health",
            "finance",
        ],
    },
}

# Notion Configuration
NOTION_CONFIG = {
    "api_version": "2022-06-28",
    "page_properties": {
        "title": {"type": "title"},
        "category": {"type": "select"},
        "tags": {"type": "multi_select"},
        "priority": {"type": "select"},
        "source": {"type": "rich_text"},
        "created_at": {"type": "date"},
    },
    "block_types": [
        "paragraph",
        "heading_1",
        "heading_2",
        "heading_3",
        "bulleted_list_item",
        "numbered_list_item",
        "callout",
        "code",
        "quote",
        "divider",
    ],
}

# Database Configuration
DATABASE_CONFIG = {
    "embedding_dimensions": 1536,
    "similarity_threshold": 0.8,
    "max_search_results": 10,
    "index_config": {"enable_semantic_search": True, "enable_keyword_search": True},
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "enable_agent_logging": True,
    "enable_workflow_logging": True,
}

# Monitoring Configuration
MONITORING_CONFIG = {
    "enable_metrics": True,
    "enable_tracing": True,
    "metrics_interval": 60,  # seconds
    "alert_thresholds": {
        "error_rate": 0.05,
        "response_time": 30,  # seconds
        "success_rate": 0.95,
    },
}


def get_agent_config(agent_name: str) -> Dict[str, Any]:
    """Get configuration for a specific agent."""
    return AGENT_CONFIG.get(agent_name, {})


def get_workflow_config() -> Dict[str, Any]:
    """Get workflow configuration."""
    return WORKFLOW_CONFIG


def get_content_config() -> Dict[str, Any]:
    """Get content processing configuration."""
    return CONTENT_CONFIG


def get_notion_config() -> Dict[str, Any]:
    """Get Notion configuration."""
    return NOTION_CONFIG


def get_database_config() -> Dict[str, Any]:
    """Get database configuration."""
    return DATABASE_CONFIG


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration."""
    return LOGGING_CONFIG


def get_monitoring_config() -> Dict[str, Any]:
    """Get monitoring configuration."""
    return MONITORING_CONFIG
