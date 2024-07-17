from haystack import Pipeline
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

from utils.utils import get_env_variable


def setup_haystack():
    url = get_env_variable("QDRANT_API_URL")
    api_key = get_env_variable("QDRANT_API_KEY")
    openai_api_key = get_env_variable("OPENAI_API_KEY")

    rag = Pipeline()
    document_store = QdrantDocumentStore(
        url=url,
        api_key=Secret.from_token(api_key),
        embedding_dim=1536,
        index="spd_manifesto",
    )
    retriever = QdrantEmbeddingRetriever(
        document_store=document_store,
        top_k=3,
    )
    template = """
        Du bist ein Experte für Wahlprogramme und beantwortest Fragen basierend auf den folgenden kontextbezogenen Informationen. Wenn du die Antwort nicht weißt, sag einfach, dass du es nicht weißt. Verwende maximal drei Sätze und halte die Antwort kurz.

        Frage: {{question}}

        Kontext:
        {% for document in documents %}
            {{ document.meta._node_content}}
        {% endfor %}

        Antwort:
    """

    prompt_builder = PromptBuilder(template=template)
    generator = OpenAIGenerator(
        model="gpt-3.5-turbo",
        generation_kwargs={"temperature": 0.1},
        api_key=Secret.from_token(openai_api_key),
    )
    embedder = OpenAITextEmbedder(
        api_key=Secret.from_token(openai_api_key),
    )
    rag.add_component("text_embedder", embedder)
    rag.add_component("retriever", retriever)
    rag.add_component("prompt_builder", prompt_builder)
    rag.add_component("generator", generator)
    rag.add_component("answer_builder", AnswerBuilder())

    rag.connect("text_embedder.embedding", "retriever.query_embedding")
    rag.connect("retriever", "prompt_builder.documents")
    rag.connect("prompt_builder", "generator")
    rag.connect("generator.replies", "answer_builder.replies")
    rag.connect("generator.meta", "answer_builder.meta")
    rag.connect("retriever", "answer_builder.documents")

    return rag
