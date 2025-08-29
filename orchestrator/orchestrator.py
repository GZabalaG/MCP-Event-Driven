import asyncio
import json
import logging
import uuid
from aiokafka import AIOKafkaConsumer
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
import os
import openai

openai.api_key = os.getenv("OPENAI_APIKEY")

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    force=True,
)
logger = logging.getLogger("orchestrator")

# ----------------------------
# MCP Servers & Tools
# ----------------------------
SERVER_URLS = {
    "action": "http://action_server:8001/mcp",
    "classification": "http://classification_server:8002/mcp",
    "extract_info": "http://info_extraction_server:8003/mcp",
    "vectorize": "http://vector_server:8004/mcp",
}

TOOLS = {
    "extract_info": ["initial_info_extraction", "convert_format", "extract_text"],
    "classification": ["classify_rules", "classify_llm"],
    "vectorize": ["vector_index", "vector_retrieve"],
    "action": ["notify", "archive"],
}

# ----------------------------
# Contextual queries for vector search
# ----------------------------
USE_CASE_QUERIES = {
    "paper": "The puprose of this paper",
}

# ----------------------------
# MCP Call Utility
# ----------------------------
async def call_tool(server_url: str, tool_name: str, args: dict):
    async with streamablehttp_client(server_url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            available = [t.name for t in tools.tools]
            if tool_name not in available:
                raise RuntimeError(f"Tool '{tool_name}' not available: {available}")

            result = await session.call_tool(tool_name, args)
            return result.content

# ----------------------------
# Mock Planner (with error-safe checks)
# ----------------------------
def mock_planner(use_case: str, exploration: dict):
    steps = []

    # If exploration failed, no plan can be built
    if exploration.get("status") == "error":
        return steps  

    doc_type = exploration.get("type", "").lower()
    doc_len = exploration.get("length", 0)

    if doc_type in ["eml"]:
        steps.append(("extract_info", "convert_format"))

    if doc_len > 500:
        steps.append(("vectorize", "vector_index"))
    else:
        steps.append(("extract_info", "extract_text"))

    if use_case == "paper":
        steps.append(("classification", "classify_llm"))
    elif use_case in ["pdf_invoice_categorization", "email_processing"]:
        steps.append(("classification", "classify_rules"))

    if use_case == "pdf_invoice_categorization":
        steps.append(("action", "archive"))
    else:
        steps.append(("action", "notify"))

    return steps


# ----------------------------
# Pipeline Execution (with error handling)
# ----------------------------
async def execute_pipeline(doc_path: str, use_case: str, pipeline_id: str):
    logger.info(f"[{pipeline_id}] Starting pipeline | Doc={doc_path} | UseCase={use_case}")

    # Step 1: Initial exploration
    exploration_raw = await call_tool(
        SERVER_URLS["extract_info"], "initial_info_extraction", {"path": doc_path}
    )
    exploration_json = json.loads(exploration_raw[0].text)
    logger.info(f"[{pipeline_id}] Initial info extracted: {exploration_json}")

    # If exploration failed → stop early
    if exploration_json.get("status") == "error":
        logger.error(f"[{pipeline_id}] ❌ Exploration failed: {exploration_json['error']}")
        return None  

    # Step 2: Plan pipeline
    plan = mock_planner(use_case, exploration_json)
    if not plan:
        logger.warning(f"[{pipeline_id}] ⚠️ No plan generated, stopping pipeline")
        return None

    logger.info(f"[{pipeline_id}] Pipeline plan: {plan}")

    # Step 3: Execute steps
    current_text = None
    doc_id = None
    classifcation_result = None
    for server_key, tool_name in plan:
        args = {}

        if tool_name in ["initial_info_extraction", "convert_format", "extract_text"]:
            args["path"] = doc_path
        elif tool_name == "vector_index":
            args["path"] = doc_path
        elif tool_name == "vector_retrieve":
            args["doc_id"] = doc_id
            args["query"] = USE_CASE_QUERIES.get(use_case, USE_CASE_QUERIES[use_case])
        elif tool_name in ["classify_rules", "classify_llm"]:
            args["use_case"] = use_case
            args["text"] = current_text
        elif tool_name in ["notify", "archive"]:
            args["action"] = tool_name
            args["payload"] = {
                "doc": doc_path,
                "user": "alice",
                "classification_result": classifcation_result
            }

        logger.info(f"[{pipeline_id}] Executing {server_key}.{tool_name} with args={args}")
        try:
            result_raw = await call_tool(SERVER_URLS[server_key], tool_name, args)
            result = json.loads(result_raw[0].text)
            logger.info(f"[{pipeline_id}] Result {server_key}.{tool_name}: {result}")

            if tool_name in ["extract_text", "vector_index"]:
                current_text = result.get("text", None)
            if tool_name == "vector_index":
                doc_id = result.get("doc_id", None)
            if tool_name in ["classify_rules", "classify_llm"]:
                classifcation_result = result.get("category", None)

        except Exception as e:
            logger.error(f"[{pipeline_id}] Step failed {server_key}.{tool_name}: {e}")
            break

    logger.info(f"[{pipeline_id}] ✅ Pipeline finished for {doc_path}")
    return current_text

# ----------------------------
# Kafka Consumer
# ----------------------------
async def create_consumer():
    from aiokafka.errors import KafkaConnectionError
    while True:
        try:
            consumer = AIOKafkaConsumer(
                "documents",
                bootstrap_servers="kafka:9092",
                group_id="orchestrator-group",
                auto_offset_reset="earliest",
            )
            await consumer.start()
            logger.info("✅ Connected to Kafka broker")
            return consumer
        except KafkaConnectionError:
            logger.warning("⚠️ Kafka not ready, retrying in 3s...")
            await asyncio.sleep(3)

# ----------------------------
# Main Loop
# ----------------------------
async def main():
    consumer = await create_consumer()
    try:
        async for msg in consumer:
            try:
                event = json.loads(msg.value.decode("utf-8"))
            except Exception as e:
                logger.warning(f"⚠️ Invalid message: {msg.value} ({e})")
                continue

            doc_path = event.get("path")
            use_case = event.get("use_case", "default")
            pipeline_id = str(uuid.uuid4())[:8]

            logger.info(f"[{pipeline_id}] 📥 Event: {event}")
            await execute_pipeline(doc_path, use_case, pipeline_id)

    finally:
        await consumer.stop()

if __name__ == "__main__":
    asyncio.run(main())
