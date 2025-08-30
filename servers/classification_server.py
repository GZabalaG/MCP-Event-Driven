import logging

from mcp.server.fastmcp import Context, FastMCP

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


# TODO use sampling to use client LLM and generate a prompt template for use case
@mcp.tool()
async def classify_llm(text: str, ctx: Context) -> dict:
    """Mock LLM classification."""
    await ctx.info(f"🤖 Mock-LLM classifying text: {text[:50]}...")
    result = {
        "status": "ok",
        "method": "llm",
        "category": "AI paper",
    }
    await ctx.debug(f"LLM Classification result: {result}")
    return result


def main():
    logging.getLogger(__name__).info(
        "🚀 Starting Classification MCP server on port 8002"
    )
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
