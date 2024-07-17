from collections import defaultdict
from typing import List, Optional, Union

import dspy
import numpy as np
import openai
from dsp.modules.sentence_vectorizer import BaseSentenceVectorizer
from dsp.utils import dotdict
from qdrant_client import QdrantClient

from utils.utils import get_env_variable
from vector_db.vector_db import setup_qdrant_client


class OpenAIVectorizer(BaseSentenceVectorizer):
    """
    This vectorizer uses OpenAI API to convert texts to embeddings. Changing `model` is not
    recommended. More about the model: https://openai.com/blog/new-and-improved-embedding-model/
    `api_key` should be passed as an argument or as env variable (`OPENAI_API_KEY`).
    """

    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        embed_batch_size: int = 1024,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.embed_batch_size = embed_batch_size

        self.Embedding = openai.embeddings

        if api_key:
            openai.api_key = api_key

    def __call__(self, inp_examples: List["Example"]) -> np.ndarray:
        text_to_vectorize = self._extract_text_from_examples(inp_examples)
        embeddings_list = []

        n_batches = (len(text_to_vectorize) - 1) // self.embed_batch_size + 1
        for cur_batch_idx in range(n_batches):
            start_idx = cur_batch_idx * self.embed_batch_size
            end_idx = (cur_batch_idx + 1) * self.embed_batch_size
            cur_batch = text_to_vectorize[start_idx:end_idx]

            response = self.Embedding.create(
                model=self.model,
                input=cur_batch,
            )

            cur_batch_embeddings = [cur_obj.embedding for cur_obj in response.data]

            embeddings_list.extend(cur_batch_embeddings)

        embeddings = np.array(embeddings_list, dtype=np.float32)
        return embeddings


class CustomQdrantRM(dspy.Retrieve):
    def __init__(
        self,
        qdrant_collection_name: str,
        qdrant_client: QdrantClient,
        k: int = 3,
        document_field: str = "document",
        vectorizer: Optional[BaseSentenceVectorizer] = None,
        vector_name: Optional[str] = None,
    ):
        super().__init__(k=k)
        self._collection_name = qdrant_collection_name
        self._client = qdrant_client
        self._vectorizer = vectorizer or OpenAIVectorizer()
        self._document_field = document_field
        self._vector_name = vector_name or self._get_first_vector_name()

    def forward(
        self, query_or_queries: Union[str, list[str]], k: Optional[int] = None
    ) -> dspy.Prediction:
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        queries = [q for q in queries if q]

        vectors = self._vectorizer(queries)
        passages_scores = defaultdict(float)
        passages = defaultdict(str)
        for vector in vectors:
            search_results = self._client.search(
                collection_name=self._collection_name,
                query_vector=vector,
                limit=k or self.k,
            )
            for result in search_results:

                document = result.payload["_node_content"]
                passages[document] = document
                passages_scores[document] += result.score

        sorted_passages = sorted(
            passages_scores.items(), key=lambda x: x[1], reverse=True
        )

        result = [dotdict({"long_text": passage}) for passage, _ in sorted_passages]
        return result

    def _get_first_vector_name(self) -> Optional[str]:
        vectors = self._client.get_collection(
            self._collection_name
        ).config.params.vectors

        if not isinstance(vectors, dict):
            return None

        first_vector_name = list(vectors.keys())[0]

        return first_vector_name or None


class GenerateAnswer(dspy.Signature):
    """Du bist ein Experte für Wahlprogramme und beantwortest Fragen basierend auf den folgenden kontextbezogenen Informationen. Wenn du die Antwort nicht weißt, sag einfach, dass du es nicht weißt. Verwende maximal drei Sätze und halte die Antwort kurz."""

    context = dspy.InputField(
        desc="Kontext zu der Frage basierend auf den Auszügen aus dem Wahlprogramm"
    )
    question = dspy.InputField(desc="Die Frage, die beantwortet werden soll")
    answer = dspy.OutputField(
        desc="Antwort auf die Frage basierend auf dem Kontext aus dem Wahlprogramm"
    )


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


def setup_dspy():
    openai_api_key = get_env_variable("OPENAI_API_KEY")
    llm = dspy.OpenAI(
        model="gpt-3.5-turbo",
        api_key=openai_api_key,
        temperature=0.1,
    )

    # Initialize QDrant Client
    qdrant_client = setup_qdrant_client()
    # Initialize QDrant Retriever Model
    qdrant_retriever_model = CustomQdrantRM(
        qdrant_collection_name="spd_manifesto",
        qdrant_client=qdrant_client,
        k=3,
    )

    # Configure DSPy settings
    dspy.settings.configure(lm=llm, rm=qdrant_retriever_model)

    rag = RAG(num_passages=3)
    return rag
