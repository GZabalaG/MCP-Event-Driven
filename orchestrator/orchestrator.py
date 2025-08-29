import asyncio
import json
import logging
import uuid
from aiokafka import AIOKafkaConsumer
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
import os
import json
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
    "extract_info": ["initial_info_extraction", "convert_format", "extract_text", "extract_info"],
    "classification": ["classify_rules", "classify_llm"],
    "vectorize": ["vector_index", "vector_retrieve"],
    "action": ["notify", "archive"],
}

# ----------------------------
# Predefined Prompts (per use case)
# ----------------------------
USE_CASE_PROMPTS = {
    "pdf_invoice_categorization": """
You are an orchestrator. Given the initial info extraction of a PDF invoice, 
decide the steps: convert to txt if needed, classify by rules, archive, and optionally classify with LLM if unknown type.
""",
    "email_processing": """
You are an orchestrator. Given the initial info extraction of an email, 
decide if vectorization is needed based on length, classify with LLM, and notify.
""",
    "default": """
You are an orchestrator. For a generic document, decide conversion, vectorization, classification, and actions.
"""
}

# ----------------------------
# MCP Call Utility
# ----------------------------
async def call_tool(server_url: str, tool_name: str, args: dict):
    """Call a tool on an MCP server and return its result."""
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
# Mock Planner (pretend LLM)
# ----------------------------
def mock_planner(use_case: str, exploration: dict):
    """
    Pretend we used the prompt + exploration with an LLM.
    Decide pipeline steps: convert, vectorize, extract_text, classify, action.
    """
    steps = []

    # Example logic: PDFs or DOCX may need conversion
    if exploration.get("type", "").lower() in ["pdf", "docx"]:
        steps.append(("extract_info", "convert_format"))

    # If document is long or email, vectorize
    if exploration.get("length", 0) > 500 or use_case == "email_processing":
        steps.append(("vectorize", "vector_index"))
    else:
        steps.append(("extract_info", "extract_text"))

    # Classification based on use case
    if use_case == "pdf_invoice_categorization":
        steps.append(("classification", "classify_llm"))
    elif use_case in ["email_processing", "report"]:
        steps.append(("classification", "classify_rules"))

    # Action
    if use_case == "pdf_invoice_categorization":
        steps.append(("action", "archive"))
    else:
        steps.append(("action", "notify"))

    return steps


"""
async def llm_planner(use_case: str, exploration: dict):
    '''
    Use GPT to decide the pipeline dynamically.
    Combines the predefined prompt for the use case with the extracted info.
    Returns: list of (server_key, tool_name)
    '''
    prompt = USE_CASE_PROMPTS.get(use_case, USE_CASE_PROMPTS["default"])
    llm_input = f"{prompt}\n\nDocument info: {json.dumps(exploration, indent=2)}\n\n"
    llm_input += "Return the pipeline as a JSON list of {'server': 'server_key', 'tool': 'tool_name'} objects."

    # Async call to OpenAI GPT
    response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[{"role": "user", "content": llm_input}],
        temperature=0
    )

    # Extract text content from GPT
    content = response.choices[0].message.content

    # Parse GPT output JSON
    try:
        steps = json.loads(content)
        pipeline = [(step["server"], step["tool"]) for step in steps]
    except Exception as e:
        # Fallback if parsing fails
        pipeline = []
        print(f"⚠️ Failed to parse GPT output: {e} | content: {content}")

    return pipeline
"""

# ----------------------------
# Pipeline Execution
# ----------------------------
async def execute_pipeline(doc_path: str, use_case: str, pipeline_id: str):
    logger.info(f"[{pipeline_id}] Starting pipeline | Doc={doc_path} | UseCase={use_case}")

    # Step 1: Initial exploration
    try:
        exploration = await call_tool(
            SERVER_URLS["extract_info"],
            "initial_info_extraction",
            {"path": doc_path},
        )
        logger.info(f"[{pipeline_id}] Initial info extracted: {exploration}")
    except Exception as e:
        logger.error(f"[{pipeline_id}] Initial exploration failed: {e}")
        return

    # Step 2: Plan pipeline (mock for now, LLM in future)
    plan = mock_planner(use_case, exploration)
    logger.info(f"[{pipeline_id}] Pipeline plan: {plan}")

    # Step 3: Execute steps
    current_text = None
    doc_id = None
    for server_key, tool_name in plan:
        args = {}

        # Set arguments based on tool
        if tool_name in ["initial_info_extraction", "convert_format"]:
            args["path"] = doc_path
        elif tool_name == "vector_index":
            args["path"] = doc_path
        elif tool_name == "extract_text":
            args["path"] = doc_path
        elif tool_name in ["classify_rules", "classify_llm"]:
            args["use_case"] = use_case
            args["text"] = current_text
        elif tool_name in ["notify", "archive", "take_action"]:
            args["action"] = tool_name
            args["payload"] = {"doc": doc_path, "user": "alice"}
        elif tool_name == "vector_retrieve":
            args["query"] = f"{use_case} context"
            args["doc_id"] = doc_id  # filter retrieval if needed

        logger.info(f"[{pipeline_id}] Executing {server_key}.{tool_name} with args={args}")
        try:
            result = await call_tool(SERVER_URLS[server_key], tool_name, args)
            logger.info(f"[{pipeline_id}] Result {server_key}.{tool_name}: {result}")

            # Update current_text if returned
            if tool_name in ["extract_text", "vector_index"]:
                current_text = result.get("text", None)
            if tool_name == "vector_index":
                doc_id = result.get("indexed_id", None)

        except Exception as e:
            logger.error(f"[{pipeline_id}] Step failed {server_key}.{tool_name}: {e}")
            break

    logger.info(f"[{pipeline_id}] Pipeline finished for {doc_path}")
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
