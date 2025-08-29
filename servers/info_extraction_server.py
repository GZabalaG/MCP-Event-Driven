import logging
import os
from mcp.server.fastmcp import FastMCP, Context
from pathlib import Path

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
    full_path = os.path.join("resources", path)

    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            doc_type = "pdf"
            length = 1000  # Mock, replace with real page/char count
        else:
            doc_type = "txt"
            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read()
            length = len(text)
    except Exception as e:
        await ctx.info(f"❌ Failed reading {full_path}: {e}")
        return {"status": "error", "error": str(e)}

    result = {"status": "ok", "doc": path, "type": doc_type, "length": length}
    await ctx.debug(f"Initial info result: {result}")
    return result


@mcp.tool()
async def convert_format(path: str, ctx: Context) -> dict:
    """
    Convert document format if needed (PDF -> TXT).
    Returns converted path.
    """
    await ctx.info(f"🔄 Converting document format: {path}")
    if path.lower().endswith(".pdf"):
        converted_path = path.replace(".pdf", ".txt")
        # Mock conversion: just return the path
        await ctx.debug(f"Converted {path} -> {converted_path}")
    else:
        converted_path = path
        await ctx.debug(f"No conversion needed for {path}")

    return {"status": "ok", "converted_path": converted_path}


@mcp.tool()
async def extract_info(path: str, prompt_config: dict, ctx: Context) -> dict:
    """
    Use a prompt (simulating LLM) to extract structured info.
    `prompt_config` can contain keys defining what info to extract.
    """
    await ctx.info(f"📝 Extracting info from {path} with prompt {prompt_config}")
    full_path = os.path.join("resources", path)

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        await ctx.info(f"❌ Failed reading {full_path}: {e}")
        return {"status": "error", "error": str(e)}

    # Mocked LLM extraction
    extracted_info = {key: f"mocked_value_for_{key}" for key in prompt_config.get("fields", [])}
    result = {"status": "ok", "doc": path, "extracted_info": extracted_info}
    await ctx.debug(f"Extracted info result: {result}")
    return result

@mcp.tool()
async def extract_text(path: str, ctx: Context, prompt: str = "") -> dict:
    """Return plain text of document (to be used by vectorization or LLM)."""
    await ctx.info(f"📄 Extracting full text from: {path} (prompt={prompt})")
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
    # future: apply LLM using prompt
    result = {"status": "ok", "doc": path, "text": text}
    await ctx.debug(f"Text extraction result: {result}")
    return result

def main():
    logging.getLogger(__name__).info("🚀 Starting Extract Info MCP server on port 8003")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
