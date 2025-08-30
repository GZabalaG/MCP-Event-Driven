import logging
from mcp.server.fastmcp import FastMCP, Context

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    force=True,
)

mcp = FastMCP("action", host="0.0.0.0", port=8001)


@mcp.tool()
async def notify(payload: dict, ctx: Context) -> dict:
    """Notify a user or system about some processed result."""
    await ctx.info(f"🔔 Notifying with payload={payload}")
    return {"status": "ok", "action": "notify", "payload": payload}


@mcp.tool()
async def archive(payload: dict, ctx: Context) -> dict:
    """Archive a processed document."""
    await ctx.info(f"🗄️ Archiving document with payload={payload}")
    return {"status": "ok", "action": "archive", "payload": payload}


def main():
    logging.getLogger(__name__).info("🚀 Starting Action MCP server on port 8001")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
