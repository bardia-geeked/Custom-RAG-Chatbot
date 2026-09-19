from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from pinecone import Pinecone
import os


def get_llm(provider: str, model: str, **kwargs) -> BaseChatModel:
    if provider == "Ollama":
        return ChatOllama(model=model, temperature=kwargs.get("temperature", 0.2))
    if provider == "OpenAI":
        return ChatOpenAI(model=model, api_key=kwargs.get("api_key") or os.getenv("OPENAI_API_KEY"))
    if provider == "Anthropic":
        return ChatAnthropic(model=model, api_key=kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY"))
    raise ValueError(f"Unknown LLM provider: {provider}")


def get_embedding(provider: str, model: str, **kwargs) -> Embeddings:
    if provider == "Ollama":
        return OllamaEmbeddings(model=model, base_url=kwargs.get("base_url"))
    if provider == "OpenAI":
        return OpenAIEmbeddings(model=model, api_key=kwargs.get("api_key") or os.getenv("OPENAI_API_KEY"))
    raise ValueError(f"Unknown embedding provider: {provider}")


def get_vectorstore(provider: str, embedding, **kwargs) -> VectorStore:
    if provider == "Chroma":
        return Chroma(
            persist_directory=kwargs.get("persist_directory", "chroma_db"),
            embedding_function=embedding
        )
    if provider == "Pinecone":
        pc = Pinecone(api_key=kwargs.get("api_key") or os.getenv("PINECONE_API_KEY"))
        index_name = kwargs.get("index_name", "rag-chatbot")
        return PineconeVectorStore(
            index=pc.Index(index_name),
            embedding=embedding,
            text_key="text"
        )
    raise ValueError(f"Unknown vectorstore provider: {provider}")