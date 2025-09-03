import logging
import uuid
from pathlib import Path

import fitz  # PyMuPDF
from mcp.server.fastmcp import Context, FastMCP
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
PINECONE_API_KEY = (
)
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
# Helpers
# ----------------------------
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ----------------------------
# MCP Server
# ----------------------------
mcp = FastMCP("vectorize", host="0.0.0.0", port=8004)


@mcp.tool()
async def vector_index(path: str, ctx: Context) -> dict:
    """Index a file (PDF or TXT) into Pinecone with chunking."""
    await ctx.info(f"📊 Indexing doc: {path}")

    try:
        file_path = Path(path).resolve()
        await ctx.debug(f"🔎 Checking file: {file_path}")
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        text = ""

        if ext == ".pdf":
            await ctx.debug(f"📂 Opening PDF with fitz: {file_path}")
            doc = fitz.open(file_path)
            await ctx.debug(f"📄 PDF has {doc.page_count} pages")
            for i, page in enumerate(doc, start=1):
                page_text = page.get_text()
                await ctx.debug(f"📑 Page {i} extracted, length={len(page_text)}")
                text += page_text + "\n"
        else:
            await ctx.debug(f"📂 Opening text file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        if not text.strip():
            raise ValueError("Extracted text is empty!")

        # Chunking
        chunks = chunk_text(text)
        await ctx.debug(f"✂️ Created {len(chunks)} chunks for embedding")

        # Encode and upsert chunks
        doc_id = str(uuid.uuid4())
        vectors = []
        for i, chunk in enumerate(chunks):
            vector = embedder.encode(chunk).tolist()
            chunk_id = f"{doc_id}_chunk_{i}"
            metadata = {
                "doc_id": doc_id,
                "chunk_id": i,
                "text": chunk,
                "source": str(file_path),
            }
            vectors.append((chunk_id, vector, metadata))

        index.upsert(vectors)
        result = {
            "status": "ok",
            "doc_id": doc_id,
            "chunks": len(chunks),
            "source": str(file_path),
        }
        await ctx.debug(f"✅ Vector index result: {result}")
        return result

    except Exception as e:
        await ctx.info(f"❌ Failed reading {path}: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def vector_retrieve(
    doc_id: str, query: str, top_k: int = 5, ctx: Context = None
) -> dict:
    """Retrieve vectors from Pinecone by doc_id and query."""
    await ctx.info(
        f"🔍 Retrieving top-{top_k} vectors for doc_id={doc_id} | query={query}"
    )

    try:
        query_vector = embedder.encode(query).tolist()
        await ctx.debug(f"⚙️ Encoded query vector, dim={len(query_vector)}")

        query_result = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter={"doc_id": {"$eq": str(doc_id).strip()}},
        )

        matches = [
            {
                "id": m["id"],
                "score": m["score"],
                "source": m["metadata"].get("source"),
                "chunk_id": m["metadata"].get("chunk_id"),
                "text": m["metadata"].get("text"),  # 👈 include retrieved text
            }
            for m in query_result.get("matches", [])
        ]

        result = {
            "status": "ok",
            "matches": matches,
            "texts": [
                m["metadata"]["text"]
                for m in query_result.get("matches", [])
                if "metadata" in m and "text" in m["metadata"]
            ],
        }

        await ctx.debug(f"✅ Vector retrieve result: {result}")
        return result

    except Exception as e:
        await ctx.info(f"❌ Retrieval failed: {e}")
        return {"status": "error", "error": str(e)}


def main():
    logging.getLogger(__name__).info("🚀 Starting Vectorize MCP server on port 8004")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
