# ReMind Knowledge Processor - Multi-Agent System

This directory contains the multi-agent system built using Google's Agent Development Kit (ADK) for processing and structuring knowledge content, creating Notion pages, and managing knowledge bases.

## Architecture Overview

The system consists of four specialized agents working together to process knowledge:

### 1. Data Structurer Agent (`data_structurer_agent.py`)

- **Purpose**: Analyzes and structures incoming content
- **Capabilities**:
  - Content analysis and categorization
  - Entity extraction and relationship mapping
  - Summary generation and key point extraction
  - Content quality validation
- **Tools**:
  - `analyze_content_structure`: Basic content analysis
  - `categorize_content`: Intelligent content categorization
  - `extract_entities_and_relations`: Named entity recognition
  - `generate_content_summary`: Executive summary generation

### 2. Notion Builder Agent (`notion_builder_agent.py`)

- **Purpose**: Creates and manages Notion pages and databases
- **Capabilities**:
  - Structured Notion page creation
  - Rich content formatting with blocks
  - Metadata and property management
  - Database organization
- **Tools**:
  - `create_notion_page`: Create new pages with metadata
  - `add_content_blocks`: Add structured content blocks
  - `update_notion_page`: Update existing pages
  - `create_notion_database`: Create organizing databases

### 3. Knowledge Processor Agent (`knowledge_processor_agent.py`)

- **Purpose**: Handles embeddings, database operations, and knowledge management
- **Capabilities**:
  - Embedding generation for semantic search
  - Database storage and retrieval
  - Knowledge quality validation
  - Similar content search
- **Tools**:
  - `generate_embeddings`: OpenAI embedding generation
  - `store_knowledge_in_database`: Database storage with metadata
  - `search_similar_knowledge`: Semantic search capabilities
  - `validate_knowledge_quality`: Quality assessment
  - `archive_knowledge`: Knowledge lifecycle management

### 4. Coordinator Agent (`coordinator_agent.py`)

- **Purpose**: Orchestrates the entire workflow
- **Capabilities**:
  - Workflow coordination and state management
  - Error handling and recovery
  - Performance monitoring and optimization
  - Sub-agent delegation
- **Tools**:
  - `process_knowledge_workflow`: Main workflow orchestration
  - `validate_workflow_input`: Input validation
  - `get_workflow_status`: Status monitoring
  - `handle_workflow_error`: Error handling
  - `optimize_workflow_performance`: Performance optimization

## Workflow Process

1. **Input Validation**: Coordinator validates incoming content and metadata
2. **Data Structuring**: Data structurer analyzes and categorizes content
3. **Notion Creation**: Notion builder creates structured pages
4. **Knowledge Processing**: Knowledge processor generates embeddings and stores data
5. **Finalization**: Coordinator ensures all steps completed successfully

## Usage

### Basic Usage

```python
from src.agents.coordinator_agent import coordinator_agent

# Process knowledge content
content = "Your knowledge content here..."
metadata = {
    "title": "Knowledge Title",
    "category": "technology",
    "tags": ["ai", "machine-learning"]
}

# Run the workflow
result = coordinator_agent.tools[0].func(content, metadata)
```

### Using the Workflow Runner

```python
from src.run_workflow import run_knowledge_workflow

# Async workflow execution
result = await run_knowledge_workflow(content, metadata)
```

### Batch Processing

```python
from src.run_workflow import process_batch_content

content_batch = [
    {"content": "Content 1", "metadata": {...}},
    {"content": "Content 2", "metadata": {...}}
]

results = await process_batch_content(content_batch)
```

## Configuration

The system is configured through `config.py` with settings for:

- **Agent Configuration**: Model settings, capabilities, thresholds
- **Workflow Configuration**: Steps, error handling, performance
- **Content Configuration**: Formats, validation rules, categorization
- **Notion Configuration**: API settings, page properties, block types
- **Database Configuration**: Embedding settings, search parameters
- **Logging Configuration**: Log levels, formats, monitoring

## Environment Variables

Required environment variables:

```bash
OPENAI_API_KEY=your_openai_api_key
NOTION_API_KEY=your_notion_api_key
NOTION_DEFAULT_PARENT_ID=your_notion_parent_page_id
GOOGLE_API_KEY=your_google_api_key  # For Gemini models
```

## Installation

1. Install dependencies:

```bash
poetry install
```

2. Set up environment variables in `.env` file

3. Run the workflow:

```bash
python src/run_workflow.py
```

## ADK Integration

The system follows ADK conventions:

- **Agent Discovery**: `agent.py` defines the root agent
- **Tool Integration**: Each agent has specialized tools
- **Sub-agent Coordination**: Coordinator manages sub-agents
- **Error Handling**: Comprehensive error handling and recovery

## Extending the System

### Adding New Agents

1. Create agent file in `src/agents/`
2. Define tools and capabilities
3. Add to coordinator's sub-agents list
4. Update configuration

### Adding New Tools

1. Define tool function
2. Add to agent's tools list
3. Update agent instructions
4. Test integration

### Custom Workflows

1. Modify coordinator agent
2. Add new workflow steps
3. Update configuration
4. Test end-to-end

## Monitoring and Logging

The system includes comprehensive logging and monitoring:

- **Agent-level logging**: Individual agent activities
- **Workflow-level logging**: Overall process tracking
- **Error tracking**: Detailed error information
- **Performance metrics**: Response times, success rates

## Best Practices

1. **Error Handling**: Always handle errors gracefully
2. **Validation**: Validate inputs at each step
3. **Logging**: Use structured logging for debugging
4. **Configuration**: Use configuration files for flexibility
5. **Testing**: Test individual agents and full workflows
6. **Monitoring**: Monitor performance and error rates

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure ADK is installed and configured
2. **API Errors**: Check API keys and permissions
3. **Workflow Failures**: Check logs for specific error details
4. **Performance Issues**: Review configuration and optimize

### Debug Mode

Enable debug logging by setting log level to DEBUG in configuration.

## Contributing

1. Follow the existing code structure
2. Add comprehensive logging
3. Include error handling
4. Update documentation
5. Test thoroughly

## License

This project follows the same license as the main ReMind Knowledge Processor.
