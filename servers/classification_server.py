import logging
from mcp.server.fastmcp import FastMCP, Context

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
    "invoice": {
        "utility_invoice": ["electricity", "water", "gas", "utility bill"],
        "purchase_invoice": ["purchase order", "invoice number", "total amount"],
        "subscription_invoice": ["subscription", "monthly fee", "plan"],
    },
    "email": {
        "personal_email": ["dear", "hi", "hello"],
        "work_email": ["subject:", "regards", "team"],
    },
    "report": {
        "financial_report": ["balance sheet", "profit and loss", "financial statement"],
        "analysis_report": ["analysis", "summary", "metrics"],
    },
}

@mcp.tool()
async def classify_rules(text: str, use_case: str, ctx: Context) -> dict:
    """Classify a document into subcategories using predefined rules."""
    await ctx.info(f"📂 Rule-based classification on: {text[:50]}...")

    category = "unknown"
    subcategory = "unknown"

    # Ensure the use case exists
    rules_for_use_case = SUBCATEGORY_RULES.get(use_case, {})
    for subcat, keywords in rules_for_use_case.items():
        if any(kw.lower() in text.lower() for kw in keywords):
            category = use_case
            subcategory = subcat
            break

    result = {
        "status": "ok",
        "method": "rules",
        "category": category,
        "subcategory": subcategory
    }
    await ctx.debug(f"Classification result: {result}")
    return result

@mcp.tool()
async def classify_llm(text: str, ctx: Context) -> dict:
    """Mock LLM classification."""
    await ctx.info(f"🤖 Mock-LLM classifying text: {text[:50]}...")
    result = {
        "status": "ok",
        "method": "llm",
        "category": "general_doc",
        "subcategory": "general"
    }
    await ctx.debug(f"LLM Classification result: {result}")
    return result

def main():
    logging.getLogger(__name__).info("🚀 Starting Classification MCP server on port 8002")
    mcp.run(transport="streamable-http")

if __name__ == "__main__":
    main()
