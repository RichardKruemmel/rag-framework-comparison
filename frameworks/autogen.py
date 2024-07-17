# Implementation of Autogen framework was explored but with the experimental design of the project, it was not possible to produce a working prototype.
# The code below is a sample of how the Autogen framework could be used to create a chatbot that retrieves information from a Qdrant database.
# Afterwards the required dependecies were removed from the project.
""" from typing import Dict, List, Union

from autogen import get_config_list
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from chromadb.utils import embedding_functions
from qdrant_client.http.models import SearchRequest, Filter, FieldCondition, MatchValue
from utils.utils import get_env_variable
from vector_db.vector_db import setup_qdrant_client

# Define the OpenAI embedding function

def get_embedding_function():
    api_key = get_env_variable("OPENAI_API_KEY")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-ada-002",
    )
    return openai_ef


class CustomQdrantRetrieveUserProxyAgent(RetrieveUserProxyAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def query_vector_db(
        self,
        query_texts: List[str],
        n_results: int = 10,
        search_string: str = "",
        **kwargs,
    ) -> Dict[str, Union[List[str], List[List[str]]]]:
        # Get embeddings from OpenAI
        embedding_function = get_embedding_function()
        embed_response = embedding_function(query_texts)

        all_embeddings: List[List[float]] = []

        for item in embed_response:
            all_embeddings.append(item)

        search_queries: List[SearchRequest] = []
        for embedding in all_embeddings:
            search_queries.append(
                SearchRequest(
                    vector=embedding,
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="group_id",
                                match=MatchValue(value=367),
                            )
                        ]
                    ),
                    limit=n_results,
                    with_payload=True,
                )
            )

        client = setup_qdrant_client()
        search_response = client.search_batch(
            collection_name="election_programs",
            requests=search_queries,
        )

        results = []
        for batch in search_response:
            batch_results = []
            for scored_point in batch:
                result = {
                    "id": scored_point.id,
                    "score": scored_point.score,
                    "payload": scored_point.payload,
                }
                batch_results.append(result)
            results.append(batch_results)

        return results

    def retrieve_docs(
        self, problem: str, n_results: int = 20, search_string: str = "", **kwargs
    ):
        results = self.query_vector_db(
            query_texts=[problem],
            n_results=n_results,
            search_string=search_string,
            **kwargs,
        )
        self._results = results
        return results


def setup_autogen():
    api_key = get_env_variable("OPENAI_API_KEY")
    config_list = [
        {
            "model": "gpt-3.5-turbo",
            "api_type": "openai",
            "api_key": api_key,
        }
    ]

    assistant = RetrieveAssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        human_input_mode="NEVER",
        llm_config={
            "seed": 42,
            "config_list": config_list,
        },
    )

    ragproxyagent = CustomQdrantRetrieveUserProxyAgent(
        name="ragproxyagent",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
        code_execution_config=False,
        retrieve_config={
            "task": "qa",
            "model": "gpt-3.5-turbo",
            "docs_path": None,
            "vector_db": None,
            "client": setup_qdrant_client(),
            "collection_name": "election_programs",
            "embedding_function": get_embedding_function(),
            "get_or_create": True,
            "overwrite": False,
            "update_context": True,
        },
    )
    assistant.reset()
    ragproxyagent.reset()

    message =  \
        Sie sind ein Experte für Wahlprogramme und helfen mir bei der Analyse.
        Bitte verwenden Sie immer die bereitgestellten Tools, um eine Frage zu beantworten. Verlassen Sie sich nicht auf Vorwissen.\
        
    qa_problem = "Um die Erderwärmung auf 1.5 Grad Celcius zu begrenzen auf wie viel Prozent soll das Minderungsziel für 2040 festgeschrieben werden?"
    ragproxyagent.initiate_chat(assistant, message=message, problem=qa_problem)
 """
