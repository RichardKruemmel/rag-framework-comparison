import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.prompts.chat import ChatPromptTemplate

from vector_db.vector_db import setup_qdrant_client


# Initialize a global dictionary to store the context for each question
context_dict = {}


def setup_vectorstore():
    qdrant_client = setup_qdrant_client()
    collection_name = "spd_manifesto"
    embedding_function = OpenAIEmbeddings()
    qdrant = Qdrant(qdrant_client, collection_name, embedding_function)
    return qdrant


def format_docs(docs):
    qdrant_client = setup_qdrant_client()
    relevant_docs = qdrant_client.retrieve(
        collection_name="spd_manifesto", ids=[doc.metadata["_id"] for doc in docs]
    )
    retrieved_passages = []
    for doc in relevant_docs:
        data = json.loads(doc.payload["_node_content"])
        retrieved_passages.append(data["text"])

    return "\n\n".join(retrieved_passages)


def inspect(state):
    """Extract and store the context passed between Runnables in LangChain and pass it on"""
    context = state["context"]
    question = state["question"]
    context_dict[question] = context
    return state


def setup_langchain():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    vectorstore = setup_vectorstore()
    retriever = vectorstore.as_retriever(k=3)

    prompt = ChatPromptTemplate.from_template(
        """Du bist ein Experte für Wahlprogramme und beantwortest Fragen basierend auf den folgenden kontextbezogenen Informationen. Wenn du die Antwort nicht weißt, sag einfach, dass du es nicht weißt. Verwende maximal drei Sätze und halte die Antwort kurz.
        Frage: {question}

        Kontext: {context}

        Antwort:
        """
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RunnableLambda(inspect)
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def setup_perf_langchain():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    vectorstore = setup_vectorstore()
    retriever = vectorstore.as_retriever(k=3)

    prompt = ChatPromptTemplate.from_template(
        """Du bist ein Experte für Wahlprogramme und beantwortest Fragen basierend auf den folgenden kontextbezogenen Informationen. Wenn du die Antwort nicht weißt, sag einfach, dass du es nicht weißt. Verwende maximal drei Sätze und halte die Antwort kurz.
        Frage: {question}

        Kontext: {context}

        Antwort:
        """
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
