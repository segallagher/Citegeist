import os
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

def embedding_model() -> Embeddings:
    """
    Returns an embedding model fron environment configuration.

    Returns:
        Embeddings: An embedding model instance from Ollama, OpenAI, or HuggingFace
    
    Raises:
        ValueError: If EMBED_MODEL not set or an unsuporrted embedding model type is specified.
    """

    model_type = os.environ.get("EMBED_MODEL_TYPE", "ollama").lower()
    if os.environ.get("EMBED_MODEL", "NULL") == "NULL":
        raise ValueError("EMBED_MODEL env var not set")

    if model_type == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=os.environ["EMBED_MODEL"])
    
    elif model_type == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=os.environ["EMBED_MODEL"])
    
    elif model_type == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=os.environ["EMBED_MODEL"])
    
    else:
        raise ValueError(f"Unsupported embedding model type: {model_type}")

def llm_model(temperature: int=None, model:str=None) -> BaseChatModel:
    """
    Returns a chat model fron environment configuration.

    Returns:
        BaseChatModel: A chat model instance from Ollama, OpenAI, or HuggingFace
    
    Raises:
        ValueError: If LLM_MODEL not set or an unsuporrted chat model type is specified.
    """

    model_type = os.environ.get("LLM_MODEL_TYPE", "ollama").lower()
    if os.environ.get("LLM_MODEL", "NULL") == "NULL":
        raise ValueError("LLM_MODEL env var not set")

    if model_type == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model if model else os.environ["LLM_MODEL"], temperature=temperature)
    
    elif model_type == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.environ["LLM_MODEL"], temperature=temperature)
    
    elif model_type == "huggingface":
        from langchain_community.chat_models import ChatHuggingFace
        return ChatHuggingFace(model_name=os.environ["LLM_MODEL"])
    
    else:
        raise ValueError(f"Unsupported chat model type: {model_type}")