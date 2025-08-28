from typing import Dict, Any
from google.adk.agents import Agent

SYSTEM_PROMPT_PAGE_DESIGNER = """
You are a Notion page designer. Your role is to:
1. Design a Notion page with appropriate formatting
2. Organize content using Notion's block system
3. Add metadata and properties to pages
4. Create databases for organizing related content
5. Update existing pages with new information

Always ensure pages are well-organized and follow Notion best practices.
Use appropriate icons, callouts, and formatting to enhance readability.
"""
# strucutre a notion page to be attractive to a user with given information


def create_notion_page(
    title: str, content: str, parent_id: str, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a new Notion page with structured content and metadata
    """
    pass


def search_current_notion_page(page_id: str) -> str:
    """
    Search the current Notion page for information
    """
    pass


page_designer_agent = Agent(
    name="page_designer",
    model="gemini-2.0-flash",
    description="Specialized agent for designing Notion pages",
    instruction=SYSTEM_PROMPT_PAGE_DESIGNER,
)
