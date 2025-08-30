import logging
from mcp.server.fastmcp import FastMCP, Context
from pathlib import Path
from email import policy
from email.parser import BytesParser

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    force=True,
)

mcp = FastMCP("extract_info", host="0.0.0.0", port=8003)


@mcp.tool()
async def initial_info_extraction(path: str, ctx: Context) -> dict:
    """
    Extract initial structured info from a document.
    Returns type and length.
    """
    await ctx.info(f"📑 Extracting initial info from: {path}")

    path = Path(path)

    try:
        ext = path.suffix.lower()
        if ext == ".pdf":
            doc_type = "pdf"
            length = 1000  # Mock, replace with real page/char count
        elif ext == ".eml":
            doc_type = "eml"
            with open(path, "rb") as f:
                msg = BytesParser(policy=policy.default).parse(f)
            body = msg.get_body(preferencelist=("plain", "html"))
            text = body.get_content() if body else ""
            length = len(text)
        else:
            doc_type = "txt"
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            length = len(text)
    except Exception as e:
        await ctx.info(f"❌ Failed reading {path}: {e}")
        return {"status": "error", "error": str(e)}

    result = {"status": "ok", "doc": str(path), "type": doc_type, "length": length}
    await ctx.debug(f"Initial info result: {result}")
    return result


@mcp.tool()
async def convert_format(path: str, ctx: Context) -> dict:
    """
    Convert document format if needed:
    - EML -> TXT
    Returns converted path.
    """
    await ctx.info(f"🔄 Converting document format: {path}")
    path = Path(path)
    ext = path.suffix.lower()
    converted_path = path

    try:
        if ext == ".eml":
            converted_path = Path(str(path).replace(".eml", ".txt"))
            with open(path, "rb") as f:
                msg = BytesParser(policy=policy.default).parse(f)
            body = msg.get_body(preferencelist=("plain", "html"))
            text = body.get_content() if body else ""
            with open(converted_path, "w", encoding="utf-8") as f:
                f.write(text)
            await ctx.debug(f"Converted {path} -> {converted_path}")
        else:
            await ctx.debug(f"No conversion needed for {path}")
    except Exception as e:
        await ctx.info(f"❌ Conversion failed: {e}")
        return {"status": "error", "error": str(e)}

    return {"status": "ok", "converted_path": str(converted_path)}


@mcp.tool()
async def extract_text(path: str, ctx: Context, prompt: str = "") -> dict:
    """Return plain text of document (to be used by vectorization or LLM)."""
    await ctx.info(f"📄 Extracting full text from: {path} (prompt={prompt})")

    path = Path(path)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    result = {"status": "ok", "doc": str(path), "text": text}
    await ctx.debug(f"Text extraction result: {result}")
    return result


def main():
    logging.getLogger(__name__).info("🚀 Starting Extract Info MCP server on port 8003")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
