"""Tests for the message processor."""

import json
import pytest
from worker.processor import MessageProcessor


class TestMessageProcessor:
    """Test cases for MessageProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = MessageProcessor()

    def test_parse_valid_message(self):
        """Test parsing a valid JSON message."""
        message_data = {
            "id": "test-123",
            "type": "knowledge.create",
            "data": {"knowledge_id": "k123", "content": "test content"},
        }
        message_body = json.dumps(message_data).encode("utf-8")

        result = self.processor._parse_message(message_body)

        assert result == message_data

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON message."""
        message_body = b"invalid json"

        with pytest.raises(ValueError, match="Invalid message format"):
            self.processor._parse_message(message_body)

    def test_validate_valid_message(self):
        """Test validating a valid message structure."""
        message_data = {
            "id": "test-123",
            "type": "knowledge.create",
            "data": {"knowledge_id": "k123"},
        }

        # Should not raise an exception
        self.processor._validate_message(message_data)

    def test_validate_missing_fields(self):
        """Test validating message with missing required fields."""
        message_data = {
            "id": "test-123",
            "type": "knowledge.create",
            # Missing "data" field
        }

        with pytest.raises(ValueError, match="Missing required field: data"):
            self.processor._validate_message(message_data)

    def test_validate_invalid_field_types(self):
        """Test validating message with invalid field types."""
        message_data = {
            "id": 123,  # Should be string
            "type": "knowledge.create",
            "data": {"knowledge_id": "k123"},
        }

        with pytest.raises(ValueError, match="Message ID must be a string"):
            self.processor._validate_message(message_data)

    def test_process_knowledge_create(self):
        """Test processing knowledge.create message."""
        message_data = {
            "id": "test-123",
            "type": "knowledge.create",
            "data": {"knowledge_id": "k123", "content": "test content"},
        }

        result = self.processor._process_knowledge_population(message_data)

        assert result["action"] == "created"
        assert result["knowledge_id"] == "k123"
        assert result["status"] == "success"

    def test_process_knowledge_update(self):
        """Test processing knowledge.update message."""
        message_data = {
            "id": "test-123",
            "type": "knowledge.update",
            "data": {"knowledge_id": "k123", "content": "updated content"},
        }

        result = self.processor._process_knowledge_population(message_data)

        assert result["action"] == "updated"
        assert result["knowledge_id"] == "k123"
        assert result["status"] == "success"

    def test_process_knowledge_delete(self):
        """Test processing knowledge.delete message."""
        message_data = {
            "id": "test-123",
            "type": "knowledge.delete",
            "data": {"knowledge_id": "k123"},
        }

        result = self.processor._process_knowledge_population(message_data)

        assert result["action"] == "deleted"
        assert result["knowledge_id"] == "k123"
        assert result["status"] == "success"

    def test_process_unknown_message_type(self):
        """Test processing unknown message type."""
        message_data = {
            "id": "test-123",
            "type": "unknown.type",
            "data": {"knowledge_id": "k123"},
        }

        with pytest.raises(ValueError, match="Unknown message type: unknown.type"):
            self.processor._process_knowledge_population(message_data)

    def test_should_retry_validation_error(self):
        """Test that validation errors should not be retried."""
        error = ValueError("Invalid message format")

        assert not self.processor.should_retry(error)

    def test_should_retry_other_error(self):
        """Test that other errors should be retried."""
        error = Exception("Network error")

        assert self.processor.should_retry(error)

    def test_retry_count_limits(self):
        """Test that retry count is respected."""
        error = Exception("Network error")

        # Should retry initially
        assert self.processor.should_retry(error)

        # Increment retry count to max
        for _ in range(3):
            self.processor.increment_retry_count()

        # Should not retry after max attempts
        assert not self.processor.should_retry(error)

    def test_reset_retry_count(self):
        """Test resetting retry count."""
        error = Exception("Network error")

        # Increment retry count
        self.processor.increment_retry_count()
        self.processor.increment_retry_count()

        # Reset retry count
        self.processor.reset_retry_count()

        # Should be able to retry again
        assert self.processor.should_retry(error)
