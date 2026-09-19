import streamlit as st
from backend.core import get_llm, get_embedding, get_vectorstore
from ingestion import ingest_url
#from langchain_core.vectorstores import VectorStoreRetriever
import asyncio

st.set_page_config(page_title="Custom RAG Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 Custom RAG Chatbot")

with st.sidebar:
    st.header("Configuration")

    llm_provider = st.selectbox("LLM Provider", ["Ollama", "OpenAI", "Anthropic"])
    llm_model = st.text_input("LLM Model", value="llama3.1" if llm_provider == "Ollama" else "gpt-4o-mini")
    llm_api_key = st.text_input("LLM API Key", type="password", help="Required for OpenAI/Anthropic")

    embed_provider = st.selectbox("Embedding Provider", ["Ollama", "OpenAI"])
    embed_model = st.text_input("Embedding Model", value="nomic-embed-text:latest" if embed_provider == "Ollama" else "text-embedding-3-small")
    embed_api_key = st.text_input("Embedding API Key", type="password", help="Required for OpenAI")

    vs_provider = st.selectbox("Vector Store", ["Chroma", "Pinecone"])
    vs_api_key = st.text_input("Vector Store API Key", type="password", help="Required for Pinecone")
    vs_index = st.text_input("Pinecone Index Name", value="rag-chatbot", help="Required for Pinecone")

    crawler_provider = st.selectbox("Crawler", ["tavily", "firecrawl"])
    crawler_api_key = st.text_input(f"{crawler_provider.capitalize()} API Key", type="password")
    crawl_depth = st.slider("Crawl Depth", 1, 5, 2)
    crawl_limit = st.slider("Max Pages", 10, 200, 50)

    chunk_size = st.number_input("Chunk Size", 500, 4000, 1000, 100)
    chunk_overlap = st.number_input("Chunk Overlap", 0, 500, 200, 50)

    if st.button("Initialize Models"):
        with st.spinner("Loading models..."):
            st.session_state.llm = get_llm(llm_provider, llm_model, api_key=llm_api_key)
            st.session_state.embedder = get_embedding(embed_provider, embed_model, api_key=embed_api_key)
            st.session_state.vectorstore = get_vectorstore(vs_provider, st.session_state.embedder, api_key=vs_api_key, index_name=vs_index)
            st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
        st.success("Models initialized!")

st.divider()

url = st.text_input("Enter URL to ingest", placeholder="https://example.com")
col1, col2 = st.columns([1, 4])
with col1:
    ingest_btn = st.button("🚀 Ingest", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ Clear Chat", use_container_width=True)

if clear_btn:
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if ingest_btn and url:
    if "embedder" not in st.session_state or "vectorstore" not in st.session_state:
        st.error("Please initialize models first in the sidebar.")
    else:
        with st.spinner(f"Crawling {url} with {crawler_provider}..."):
            crawler_config = {
                "max_depth": crawl_depth,
                "limit": crawl_limit,
                "api_key": crawler_api_key,
            }
            chunker_config = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
            try:
                count = asyncio.run(ingest_url(
                    url, crawler_provider, crawler_config, chunker_config,
                    st.session_state.embedder, st.session_state.vectorstore
                ))
                st.success(f"Ingested {count} chunks from {url}")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.caption(f"{i}. {src.metadata.get('source', 'unknown')}")
                    st.text(src.page_content[:300] + "...")

if prompt := st.chat_input("Ask a question..."):
    if "retriever" not in st.session_state:
        st.error("Please ingest a URL first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                docs = st.session_state.retriever.invoke(prompt)
                context = "\n\n".join(f"Source: {d.metadata.get('source')}\n{d.page_content}" for d in docs)

                system = f"""You are a helpful assistant. Answer using ONLY the provided context.
                If the answer isn't in the context, say "I don't know based on the provided sources."
                Always cite sources 
                Example for citing sources: Source: Prices - Live Exchange Rate

                Context:
                {context}"""

                stream = st.session_state.llm.astream([
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ])
                answer = st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": docs})