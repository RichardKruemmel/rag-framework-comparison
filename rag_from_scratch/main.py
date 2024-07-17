from openai import OpenAI
import json
from vector_db.vector_db import setup_qdrant_client


class Basic_RAG:
    def __init__(self):
        self.client = OpenAI()
        self.qdrant_client = setup_qdrant_client()

    def embedding_function(self, text):
        response = self.client.embeddings.create(
            input=text, model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def retrieve_documents(self, query_vector, limit=5):
        search_result = self.qdrant_client.search(
            collection_name="spd_manifesto",
            query_vector=query_vector,
            limit=limit,
        )
        return search_result

    def format_docs(self, docs):
        formatted_docs = []
        for doc in docs:
            content = doc.payload["_node_content"]
            transformed_content = json.loads(content)
            formatted_docs.append(transformed_content["text"])
        return formatted_docs

    def generate_response(self, query, documents):
        template = """Du bist ein Experte für Wahlprogramme und beantwortest Fragen basierend auf den folgenden kontextbezogenen Informationen. Wenn du die Antwort nicht weißt, sag einfach, dass du es nicht weißt. Verwende maximal drei Sätze und halte die Antwort kurz.
            Frage: {question}

            Kontext: {context}

            Antwort:
        """
        prompt = template.format(question=query, context="\n\n".join(documents))

        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": "Du bist ein Experte für Wahlprogramme und beantwortest Fragen basierend auf den folgenden kontextbezogenen Informationen. Wenn du die Antwort nicht weißt, sag einfach, dass du es nicht weißt. Verwende maximal drei Sätze und halte die Antwort kurz.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        return completion.choices[0].message.content

    def query(self, query):
        query_embedding = self.embedding_function(query)

        documents = self.retrieve_documents(query_embedding)

        processed_documents = self.format_docs(documents)

        response = self.generate_response(query, processed_documents)

        return {"response": response, "context": processed_documents}
