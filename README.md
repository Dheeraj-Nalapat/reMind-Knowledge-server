# ReMind Knowledge Populating Worker

A RabbitMQ worker service for populating knowledge in the ReMind system.

## Features

- RabbitMQ message consumer
- Configurable message processing
- Structured logging with Loguru
- Environment-based configuration
- Type hints and code quality tools

## Prerequisites

- Python 3.9+
- Poetry
- RabbitMQ server

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd reMind-Knowledge-populating-worker
```

2. Install dependencies using Poetry:

```bash
poetry install
```

3. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your configuration
```

## Configuration

Create a `.env` file with the following variables:

```env
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USERNAME=guest
RABBITMQ_PASSWORD=guest
RABBITMQ_VIRTUAL_HOST=/
QUEUE_NAME=knowledge_population
LOG_LEVEL=INFO
```

## Usage

### Development

Run the worker in development mode:

```bash
poetry run python -m worker.main
```

### Production

Build and run the worker:

```bash
poetry build
poetry run python -m worker.main
```

## Development

### Code Quality

- **Formatting**: `poetry run black .`
- **Linting**: `poetry run flake8 .`
- **Type Checking**: `poetry run mypy .`

### Testing

Run tests:

```bash
poetry run pytest
```

## Project Structure

```
├── worker/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── consumer.py
│   └── processor.py
├── tests/
│   └── __init__.py
├── pyproject.toml
├── README.md
└── .env.example
```

## License

[Add your license here]
