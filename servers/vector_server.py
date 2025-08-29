import logging
import os
import uuid
from mcp.server.fastmcp import FastMCP, Context
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    force=True,
)

# ----------------------------
# Pinecone Init
# ----------------------------
PINECONE_API_KEY = "pcsk_4hANZE_Pj2J5QUcPLPPiL8WXZKzBMba2es7PMLyRaett6bq9QUiswrdE7953iN4sBN5BdB"
if not PINECONE_API_KEY:
    raise RuntimeError("❌ Missing Pinecone API key")

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "documents"
DIMENSION = 384

if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    logging.info(f"Creating Pinecone index {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(INDEX_NAME)

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# MCP Server
# ----------------------------
mcp = FastMCP("vectorize", host="0.0.0.0", port=8004)

@mcp.tool()
async def vector_index(path: str, ctx: Context) -> dict:
    await ctx.info(f"📊 Indexing doc: {path}")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        await ctx.info(f"❌ Failed reading {path}: {e}")
        return {"status": "error", "error": str(e)}

    vector = embedder.encode(text).tolist()
    doc_id = str(uuid.uuid4())
    metadata = {"doc_id": doc_id, "source": path}
    index.upsert([(doc_id, vector, metadata)])

    result = {"status": "ok", "doc_id": doc_id, "source": path}
    await ctx.debug(f"Vector index result: {result}")
    return result

@mcp.tool()
async def vector_retrieve(doc_id: str, query: str, top_k: int = 5, ctx: Context = None) -> dict:
    await ctx.info(f"🔍 Retrieving top-{top_k} vectors for doc_id={doc_id} | query={query}")
    
    # Here, you can extend with real embedding of query if needed
    # For mock purposes, we'll just filter by doc_id metadata
    query_result = index.query(top_k=top_k, include_metadata=True, filter={"doc_id": {"$eq": doc_id}})
    
    matches = [
        {"id": m["id"], "score": m["score"], "source": m["metadata"].get("source")}
        for m in query_result["matches"]
    ]
    result = {"status": "ok", "matches": matches}
    await ctx.debug(f"Vector retrieve result: {result}")
    return result

def main():
    logging.getLogger(__name__).info("🚀 Starting Vectorize MCP server on port 8004")
    mcp.run(transport="streamable-http")

if __name__ == "__main__":
    main()
