import json
from kafka import KafkaProducer

# Kafka broker inside Docker network
BROKER = "localhost:29092"  # use 'kafka:9092' if running from docker
TOPIC = "documents"

print("Generating events...")

producer = KafkaProducer(
    bootstrap_servers=BROKER, value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# Example events with proper use_case and paths
events = [
    # {"use_case": "pdf_invoice_categorization", "path": "resources/test_docs/invoice1.txt"},
    # {"use_case": "pdf_invoice_categorization", "path": "resources/test_docs/invoice2.txt"},
    # {"use_case": "email_processing", "path": "resources/test_docs/email1.eml"},
    # {"use_case": "email_processing", "path": "resources/test_docs/email2.eml"},
    {"use_case": "paper", "path": "resources/test_docs/paper1.pdf"},
]

for event in events:
    producer.send(TOPIC, event)
    print(f"Sent: {event}")

producer.flush()
print("✅ All test events sent")
