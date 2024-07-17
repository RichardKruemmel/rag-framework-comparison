import logging
from qdrant_client import QdrantClient
from utils.utils import get_env_variable


def get_vector_store_credentials() -> str:
    try:
        qdrant_api_key = get_env_variable("QDRANT_API_KEY")
        qdrant_url = get_env_variable("QDRANT_API_URL")
        return qdrant_api_key, qdrant_url
    except Exception as e:
        logging.error(f"An error occurred while reading the credentials: {e}")
        raise


def setup_qdrant_client() -> QdrantClient:
    try:
        qdrant_api_key, qdrant_url = get_vector_store_credentials()
        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        return qdrant_client
    except Exception as e:
        logging.error(f"An error occurred while setting up the Qdrant client: {e}")
        raise
