"""This file has been copied from the Wahlwave project to show how it ingested data from PDFs in to a Vector Store."""

import os
from typing import Any, List
from llama_index import Document
from llama_index.ingestion import IngestionPipeline, IngestionCache
from llama_index.ingestion.cache import RedisCache
from llama_index.text_splitter import SentenceSplitter
from llama_index.extractors import (
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    KeywordExtractor,
)
from llama_index.vector_stores import QdrantVectorStore
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
import logging
import requests
from llama_index.llm import setup_llm_35

from llama_index.vector_store import setup_vector_store
from llama_index.templates import (
    SUMMARY_EXTRACT_TEMPLATE,
    QUESTION_GEN_TEMPLATE,
)
from scraper.utils.pdf_downloader import download_pdf
from llama_index.loader import load_docs
from utils.utils import get_env_variable


def setup_ingestion_pipeline(vector_store: QdrantVectorStore):
    azure_endpoint = get_env_variable("AZURE_OPENAI_ENDPOINT")
    api_key = get_env_variable("AZURE_OPENAI_API_KEY")
    api_version = get_env_variable("OPENAI_API_VERSION")
    llm_35 = setup_llm_35()
    transformations = [
        SentenceSplitter(chunk_size=512),
        AzureOpenAIEmbedding(
            azure_deployment="wahlwave-embedding",
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            model="text-embedding-ada-002",
            embed_batch_size=16,
        ),
        QuestionsAnsweredExtractor(llm_35, prompt_template=QUESTION_GEN_TEMPLATE),
        SummaryExtractor(llm_35, prompt_template=SUMMARY_EXTRACT_TEMPLATE),
        KeywordExtractor(llm_35, num_keywords=3),
    ]

    try:
        pipeline = IngestionPipeline(
            transformations=transformations,
            vector_store=vector_store,
            cache=IngestionCache(
                cache=RedisCache(), collection="election_program_cache"
            ),
        )
        logging.info("Ingestion pipeline successfully set up.")
        return pipeline
    except Exception as e:
        logging.error(f"An error occurred while setting up the ingestion pipeline: {e}")
        raise


def ingest_data(election_programs: List[Any]):
    vector_store = setup_vector_store("election_programs")
    for program in election_programs:
        # Define the path where the PDF will be saved
        save_path = os.path.join("./docs", f"{program.election_id}_{program.id}.pdf")
        logging.info(f"Downloading PDF for election_program_id={program.id}")

        # Download the PDF
        download_pdf(program.file_cloud_url, save_path)
        logging.info(f"Downloaded PDF to {save_path}")

        documents = load_docs(save_path, program)

        # Setup and run the ingestion pipeline for the downloaded PDF
        ingestion_pipeline = setup_ingestion_pipeline(vector_store=vector_store)
        try:
            if len(documents) == 0:
                raise ValueError("No document provided for ingestion.")
            index = ingestion_pipeline.run(documents=documents)
            logging.info(f"Document successfully ingested.")
        except Exception as e:
            logging.error(f"An error occurred while ingesting Document: {e}")
            raise

        # Delete the PDF
        os.remove(save_path)
        logging.info(f"Deleted PDF from {save_path}")
