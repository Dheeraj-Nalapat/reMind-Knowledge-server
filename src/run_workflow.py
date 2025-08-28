"""
Workflow runner for the ReMind Knowledge Processor multi-agent system.
This script demonstrates how to use the agents to process knowledge content.
"""

import asyncio
import json
from typing import Dict, Any
from src.agents.coordinator_agent import coordinator_agent
from src.common.logger.logger import get_logger

logger = get_logger(__name__)


async def run_knowledge_workflow(
    content: str, metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Run the complete knowledge processing workflow using the multi-agent system.
    """
    try:
        logger.info("Starting knowledge processing workflow")

        # Initialize metadata if not provided
        if metadata is None:
            metadata = {
                "title": "Knowledge Entry",
                "source": "manual",
                "category": "general",
                "tags": [],
            }

        # Run the workflow using the coordinator agent
        # Note: This is a simplified example. In practice, you would use ADK's runner
        workflow_result = coordinator_agent.tools[0].func(content, metadata)

        logger.info("Knowledge processing workflow completed")
        return workflow_result

    except Exception as e:
        logger.error(f"Error running knowledge workflow: {e}")
        raise


async def process_batch_content(content_batch: list) -> list:
    """
    Process a batch of content items using the multi-agent system.
    """
    try:
        results = []

        for i, item in enumerate(content_batch):
            logger.info(f"Processing item {i+1}/{len(content_batch)}")

            content = item.get("content", "")
            metadata = item.get("metadata", {})

            result = await run_knowledge_workflow(content, metadata)
            results.append(result)

        logger.info(f"Batch processing completed: {len(results)} items processed")
        return results

    except Exception as e:
        logger.error(f"Error processing batch content: {e}")
        raise


def main():
    """
    Main function to demonstrate the workflow.
    """
    # Example content to process
    example_content = """
    # Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed.
    
    ## Key Concepts:
    - Supervised Learning: Learning from labeled data
    - Unsupervised Learning: Finding patterns in unlabeled data
    - Reinforcement Learning: Learning through interaction with environment
    
    ## Applications:
    - Image recognition
    - Natural language processing
    - Recommendation systems
    - Autonomous vehicles
    """

    example_metadata = {
        "title": "Introduction to Machine Learning",
        "source": "educational",
        "category": "technology",
        "tags": ["machine-learning", "ai", "education"],
        "priority": "high",
    }

    # Run the workflow
    try:
        result = asyncio.run(run_knowledge_workflow(example_content, example_metadata))
        print("Workflow Result:")
        print(json.dumps(result, indent=2))

    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
