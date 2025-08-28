"""
Main agent file for the ReMind Knowledge Processor multi-agent system.
This file defines the root agent that coordinates the entire workflow.
"""

from src.agents.coordinator_agent import coordinator_agent
from src.common.logger.logger import get_logger

logger = get_logger(__name__)

# Define the root agent for ADK discovery
root_agent = coordinator_agent

logger.info("ReMind Knowledge Processor multi-agent system initialized")
logger.info(f"Root agent: {root_agent.name}")
logger.info(f"Sub-agents: {[agent.name for agent in root_agent.sub_agents]}")
