import asyncio
import json
import logging
import os
import uuid

from openai import AsyncOpenAI
from aiokafka import AIOKafkaConsumer
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from mcp.types import SamplingMessage, TextContent

import mcp.types as types
from mcp.shared.context import RequestContext
from typing import Any

async_openai = AsyncOpenAI(
    api_key="sk-xxx"
)

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

# ----------------------------
# Contextual queries for vector search
# ----------------------------
USE_CASE_QUERIES = {
    "paper": "The puprose of this paper",
}

# ----------------------------
# Sampling Handler
# ----------------------------
async def sampling_handler_mock(
    context: RequestContext["ClientSession", Any],
    params: types.CreateMessageRequestParams,
) -> types.CreateMessageResult:
    logger.info(f"🧠 [sampling_handler] Received context: {context}")
    logger.info(f"🧠 [sampling_handler] Received params: {params}")

    # Build a single TextContent object directly
    text_content = types.TextContent(type="text", text="MOCK result_text")

    # Return a proper CreateMessageResult
    result = types.CreateMessageResult(
        role="assistant",           # REQUIRED
        model="mock-model",         # REQUIRED
        content=text_content      # List of Content instances
    )

    return result


async def sampling_handler(
    context: RequestContext["ClientSession", Any],
    params: types.CreateMessageRequestParams,
) -> types.CreateMessageResult:
    logger.info(f"🧠 [sampling_handler] Received context: {context}")
    logger.info(f"🧠 [sampling_handler] Received params: {params}")

    # Convert MCP SamplingMessage to OpenAI chat format
    openai_messages = [
        {"role": m.role, "content": m.content.text if isinstance(m.content, types.TextContent) else str(m.content)}
        for m in params.messages
    ]

    # Provide defaults if parameters are None
    temperature = params.temperature if params.temperature is not None else 0.2
    max_tokens = params.maxTokens if params.maxTokens is not None else 200
    stop = params.stopSequences if params.stopSequences else None

    # Call OpenAI
    response = await async_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=openai_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop
    )

    # Extract text
    assistant_text = response.choices[0].message.content.strip()

    # Return a proper MCP result
    return types.CreateMessageResult(
        role="assistant",
        model=response.model,
        content=[types.TextContent(type="text", text=assistant_text)]
    )


# ----------------------------
# MCP Call Utility
# ----------------------------
async def call_tool(server_url: str, tool_name: str, args: dict):
    async with streamablehttp_client(server_url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream, sampling_callback=sampling_handler) as session:
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
        steps.append(("vectorize", "vector_retrieve"))
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

    logger.info(f"Plan response:\n{steps}")

    return steps

# ----------------------------
# LLM Planner using available MCP tools
# ----------------------------
LLM_PROMPT_TEMPLATE = """
You are an expert workflow planner for processing company documents.

Document type: {doc_type}
Document length: {doc_len}
Business use case: {use_case}

Available tools (server_key.tool_name):
{available_tools}

We want to process documents in the following structured order:

1. Conversion:
   - If the document is an email (eml), convert it to txt.
   - Otherwise, skip conversion.

2. Indexing:
   - If document length > 500 characters, store it in a vector index for later retrieval.
   - Otherwise skip indexing.

3. Classification (per use case):
   - For 'paper', classify using an LLM.
   - For 'pdf_invoice_categorization', classify using rules.
   - For 'email_processing', classify using rules.
   - For other use cases, decide the best method automatically.

4. Action:
   - For 'pdf_invoice_categorization', archive the document.
   - For all other cases, notify the responsible user.

Return strictly a Python list of tuples [(server_key, tool_name), ...] representing the workflow steps.
"""

async def get_all_tools(server_urls: dict) -> list[str]:
    """Fetch tools from all MCP servers."""
    all_tools = []
    for server_key, url in server_urls.items():
        async with streamablehttp_client(url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream, sampling_callback=sampling_handler) as session:
                await session.initialize()
                tools = await session.list_tools()
                all_tools.extend([f"{server_key}.{t.name}" for t in tools.tools])
    
    logger.info(f"Tools available:\n{all_tools}")
    return all_tools


async def llm_planner(use_case: str, exploration: dict, server_urls: dict) -> list[tuple[str]]:
    if exploration.get("status") == "error":
        return []

    doc_type = exploration.get("type", "unknown").lower()
    doc_len = exploration.get("length", 0)

    # Fetch tools from all servers
    available_tools = await get_all_tools(server_urls)
    tools_text = "\n".join(available_tools)

    prompt = LLM_PROMPT_TEMPLATE.format(
        doc_type=doc_type,
        doc_len=doc_len,
        use_case=use_case,
        available_tools=tools_text,
    )

    logger.info(f"Sending structured prompt to GPT:\n{prompt}")

    response = await async_openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=400,
    )

    text = response.choices[0].message.content.strip()

    try:
        plan = eval(text)
        logger.info(f"Plan response:\n{text}")
        if not isinstance(plan, list) or not all(isinstance(t, tuple) and len(t) == 2 for t in plan):
            raise ValueError("LLM returned invalid format")
        return plan
    except Exception as e:
        logger.error(f"Failed to parse LLM response: {e}")
        return []

# ----------------------------
# Pipeline Execution (with error handling)
# ----------------------------
async def execute_pipeline(doc_path: str, use_case: str, pipeline_id: str):
    logger.info(
        f"[{pipeline_id}] Starting pipeline | Doc={doc_path} | UseCase={use_case}"
    )

    # Step 1: Initial exploration
    exploration_raw = await call_tool(
        SERVER_URLS["extract_info"], "initial_info_extraction", {"path": doc_path}
    )
    exploration_json = json.loads(exploration_raw[0].text)
    logger.info(f"[{pipeline_id}] Initial info extracted: {exploration_json}")

    # If exploration failed → stop early
    if exploration_json.get("status") == "error":
        logger.error(
            f"[{pipeline_id}] ❌ Exploration failed: {exploration_json['error']}"
        )
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
                "classification_result": classifcation_result,
            }

        logger.info(
            f"[{pipeline_id}] Executing {server_key}.{tool_name} with args={args}"
        )
        try:
            result_raw = await call_tool(SERVER_URLS[server_key], tool_name, args)
            result = json.loads(result_raw[0].text)
            logger.info(f"[{pipeline_id}] Result {server_key}.{tool_name}: {result}")

            if tool_name == "extract_text":
                current_text = result.get("text", None)
            if tool_name == "vector_retrieve":
                current_text = str(result.get("matches", None))
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
