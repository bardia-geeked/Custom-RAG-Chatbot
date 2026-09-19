#import asyncio
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_tavily import TavilyCrawl
from firecrawl import FirecrawlApp


async def ingest_url(
    url: str,
    crawler: str,
    crawler_config: dict,
    chunker_config: dict,
    embedder: Embeddings,
    vectorstore: VectorStore,
) -> int:
    docs = await _crawl(url, crawler, crawler_config)
    if not docs:
        return 0

    chunks = _chunk(docs, chunker_config)
    await _index(chunks, embedder, vectorstore)
    return len(chunks)


async def _crawl(url: str, crawler: str, config: dict) -> List[Document]:
    if crawler == "tavily":
        return await _crawl_tavily(url, config)
    if crawler == "firecrawl":
        return await _crawl_firecrawl(url, config)
    raise ValueError(f"Unknown crawler: {crawler}")


async def _crawl_tavily(url: str, config: dict) -> List[Document]:
    import os
    api_key = config.get("api_key")
    if api_key:
        os.environ["TAVILY_API_KEY"] = api_key
    
    tavily = TavilyCrawl(
        max_depth=config.get("max_depth", 2),
        max_breadth=config.get("max_breadth", 10),
        limit=config.get("limit", 50),
        extract_depth=config.get("extract_depth", "advanced"),
    )
    res = tavily.invoke({
        "url": url,
        "max_depth": config.get("max_depth", 2),
        "extract_depth": config.get("extract_depth", "advanced"),
        "instructions": config.get("instructions", "crawl all pages"),
    })

    results = []
    if isinstance(res, dict):
        results = res.get("results") or res.get("results ") or []

    return [
        Document(page_content=r.get("raw_content", ""), metadata={"source": r.get("url", url)})
        for r in results
        if r.get("raw_content")
    ]


async def _crawl_firecrawl(url: str, config: dict) -> List[Document]:
    app = FirecrawlApp(api_key=config.get("api_key"))
    res = app.crawl_url(
        url,
        params={
            "limit": config.get("limit", 50),
            "maxDepth": config.get("max_depth", 2),
            "scrapeOptions": {"formats": ["markdown"]},
        }
    )

    docs = []
    for page in res.get("data", []):
        content = page.get("markdown") or page.get("content")
        if content:
            docs.append(Document(page_content=content, metadata={"source": page.get("url", url)}))
    return docs


def _chunk(docs: List[Document], config: dict) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.get("chunk_size", 1000),
        chunk_overlap=config.get("chunk_overlap", 200),
    )
    return splitter.split_documents(docs)


async def _index(chunks: List[Document], embedder: Embeddings, vectorstore: VectorStore):
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        await vectorstore.aadd_documents(batch)