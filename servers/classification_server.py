import logging
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from mcp.types import SamplingMessage, TextContent

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    force=True,
)

mcp = FastMCP("classification", host="0.0.0.0", port=8002)

# ----------------------------
# Subcategory rule sets
# ----------------------------
SUBCATEGORY_RULES = {
    "pdf_invoice_categorization": {
        "utility_invoice": ["electricity", "water", "gas", "utility bill"],
        "purchase_invoice": ["purchase order", "invoice number", "total amount"],
        "subscription_invoice": ["subscription", "monthly fee", "plan"],
    },
    "email_processing": {
        "personal_email": ["dear", "hi"],
        "work_email": ["subject:", "regards", "team"],
    },
}

# ----------------------------
# LLM prompts per use case
# ----------------------------
LLM_CLASSIFICATION_PROMPTS = {
    "paper": "Classify this document into one of the following categories: AI Paper, Bio Paper, Law Paper."
}


@mcp.tool()
async def classify_rules(text: str, use_case: str, ctx: Context) -> dict:
    """Classify a document into subcategories using predefined rules."""
    await ctx.info(f"📂 Rule-based classification on: {text[:50]}...")

    category = "unknown"

    # Ensure the use case exists
    rules_for_use_case = SUBCATEGORY_RULES.get(use_case, {})
    for cat, keywords in rules_for_use_case.items():
        if any(kw.lower() in text.lower() for kw in keywords):
            category = cat
            break

    result = {
        "status": "ok",
        "method": "rules",
        "category": category,
    }
    await ctx.debug(f"Classification result: {result}")
    return result


@mcp.tool()
async def classify_llm(text: str, use_case: str, ctx: Context[ServerSession, None]) -> dict:
    """Classify a document using an LLM via sampling."""
    prompt = LLM_CLASSIFICATION_PROMPTS.get(use_case, "Classify this document into an appropriate category.")
    full_prompt = f"{prompt}\n\nDocument:\n{text}"

    await ctx.info(f"🤖 LLM classifying text (use_case={use_case}): {text[:50]}...")

    # Create a sampling message
    result_msg = await ctx.session.create_message(
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text=full_prompt)
            )
        ],
        max_tokens=100,
    )

    category = "unknown"
    if result_msg.content.type == "text":
        category = result_msg.content.text.strip()

    result = {
        "status": "ok",
        "method": "llm",
        "category": category,
    }

    await ctx.debug(f"LLM Classification result: {result}")
    return result


def main():
    logging.getLogger(__name__).info("🚀 Starting Classification MCP server on port 8002")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
