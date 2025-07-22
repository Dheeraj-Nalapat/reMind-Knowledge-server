import pika
import json
from src.config import Config
from src.common.logger.logger import get_logger
from src.process_knowledge import process_content_entry

logger = get_logger(__name__)

def start_worker():
    logger.info("Connecting to RabbitMQ...")
    connection = pika.BlockingConnection(pika.URLParameters(Config.RABBITMQ_URL))
    channel = connection.channel()

    queue_name = 'knowledge_queue'
    channel.queue_declare(queue=queue_name, durable=True)

    def callback(ch, method, properties, body):
        logger.info("Received a message.")
        try:
            entry = json.loads(body.decode())
            process_content_entry(entry)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info("Processed and acknowledged message.")
        except Exception as e:
            logger.exception("Failed to process message")

    logger.info("Worker is listening for messages on queue: %s", queue_name)
    channel.basic_consume(queue=queue_name, on_message_callback=callback)
    channel.start_consuming()

if __name__ == "__main__":
    start_worker()