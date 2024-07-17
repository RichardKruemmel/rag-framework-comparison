import logging
from llama_index.core import PromptTemplate, Settings, get_response_synthesizer
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore

from utils.utils import get_env_variable
from vector_db.vector_db import setup_qdrant_client


def setup_vector_store(collection_name: str) -> QdrantVectorStore:
    try:
        qdrant_client = setup_qdrant_client()
        vector_store = QdrantVectorStore(
            client=qdrant_client, collection_name=collection_name
        )
        return vector_store
    except Exception as e:
        logging.error(f"An error occurred while setting up the vector store: {e}")
        raise


def setup_index():
    vector_store = setup_vector_store("spd_manifesto")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index


def setup_gpt_35():
    openai_api_key = get_env_variable("OPENAI_API_KEY")
    gpt_35 = OpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0.1)
    return gpt_35


def setup_ada_2_gpt():
    openai_api_key = get_env_variable("OPENAI_API_KEY")
    ada_2_gpt = OpenAIEmbedding(api_key=openai_api_key, model="text-embedding-ada-002")
    return ada_2_gpt


def setup_llama_index():
    Settings.llm = setup_gpt_35()
    Settings.embed_model = setup_ada_2_gpt()

    vector_index = setup_index()
    retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=3,
    )

    qa_prompt_tmpl = (
        "Du bist ein Experte für Wahlprogramme und beantwortest Fragen basierend auf den folgenden kontextbezogenen Informationen. Wenn du die Antwort nicht weißt, sag einfach, dass du es nicht weißt. Verwende maximal drei Sätze und halte die Antwort kurz.\n"
        "---------------------\n"
        "Frage: {query_str}\n"
        "---------------------\n"
        "Kontext:\n"
        "{context_str}\n"
        "---------------------\n"
        "Antwort: "
    )
    prompt = PromptTemplate(qa_prompt_tmpl)
    response_synthesizer = get_response_synthesizer(text_qa_template=prompt)

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    return query_engine
