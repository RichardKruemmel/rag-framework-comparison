import json

from dotenv import load_dotenv

from frameworks.crewai import setup_crew
from frameworks.dspy import setup_dspy
from frameworks.haystack import setup_haystack
from frameworks.langchain import context_dict, setup_langchain
from frameworks.llama_index import setup_llama_index
from rag_from_scratch.main import Basic_RAG


def generate_langchain_dataset():
    with open("eval/updated_qa_pairs.json", "r") as file:
        qa_pairs = json.load(file)

    rag_chain = setup_langchain()

    dataset = []

    for pair in qa_pairs:
        question = pair["question"]
        ground_truth = pair["answer"]

        response = rag_chain.invoke(question)

        data = {
            "question": question,
            "answer": response,
            "contexts": [],  # This will be filled later
            "ground_truths": ground_truth,
        }

        dataset.append(data)

    for data in dataset:
        question = data["question"]
        data["contexts"] = context_dict.get(question, [])

    with open("dataset/langchain_dataset1.json", "w") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)


def generate_haystack_dataset():

    with open("eval/updated_qa_pairs.json", "r") as file:
        qa_pairs = json.load(file)

    rag = setup_haystack()

    dataset = []

    for pair in qa_pairs:
        question = pair["question"]
        ground_truth = pair["answer"]

        response = rag.run(
            {
                "text_embedder": {"text": question},
                "prompt_builder": {"question": question},
                "answer_builder": {"query": question},
            }
        )

        answer = response["answer_builder"]["answers"][0].data

        documents = response["answer_builder"]["answers"][0].documents
        context = []
        for document in documents:
            document_data = document.to_dict()
            content = json.loads(document_data["_node_content"])
            context.append(content["text"])

        data = {
            "question": question,
            "answer": answer,
            "contexts": context,
            "ground_truths": ground_truth,
        }

        dataset.append(data)

    with open("dataset/haystack_dataset1.json", "w") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)


def generate_dspy_dataset():
    with open("eval/updated_qa_pairs.json", "r") as file:
        qa_pairs = json.load(file)

    rag = setup_dspy()

    dataset = []

    for pair in qa_pairs:
        question = pair["question"]
        ground_truth = pair["answer"]

        response = rag(question)

        answer = response.answer

        documents = response.context
        context = []
        for document in documents:
            data = json.loads(document)
            context.append(data["text"])

        data = {
            "question": question,
            "answer": answer,
            "contexts": context,
            "ground_truths": ground_truth,
        }

        dataset.append(data)

    with open("dataset/dspy_dataset1.json", "w") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)


def generate_llama_dataset():
    with open("eval/updated_qa_pairs.json", "r") as file:
        qa_pairs = json.load(file)

    llama = setup_llama_index()

    dataset = []

    for pair in qa_pairs:
        question = pair["question"]
        ground_truth = pair["answer"]

        response = llama.query(question)

        source_nodes = response.source_nodes
        context = [node.text for node in source_nodes]

        data = {
            "question": question,
            "answer": str(response),
            "contexts": context,
            "ground_truths": ground_truth,
        }

        dataset.append(data)

    with open("dataset/llama_index_dataset1.json", "w") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)


def generate_crewai_dataset():

    with open("eval/updated_qa_pairs.json", "r") as file:
        qa_pairs = json.load(file)

    crew = setup_crew()
    dataset = []

    for pair in qa_pairs:
        question = pair["question"]
        ground_truth = pair["answer"]

        response = crew.kickoff(inputs={"question": question})
        data = {
            "question": question,
            "answer": response,
            "contexts": [],
            "ground_truths": ground_truth,
        }

        dataset.append(data)

    with open("dataset/crewai_dataset_without_context1.json", "w") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)

    with open("dataset/crew_ai_context1.json", "r") as file:
        context_map = json.load(file)

    for i, data in enumerate(dataset):
        data["contexts"] = context_map[i]

    with open("dataset/crewai_dataset1.json", "w") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)


def generate_basic_rag_dataset():
    with open("eval/updated_qa_pairs.json", "r") as file:
        qa_pairs = json.load(file)

    basic_rag = Basic_RAG()

    dataset = []

    for pair in qa_pairs:
        question = pair["question"]
        ground_truth = pair["answer"]

        response = basic_rag.query(question)

        data = {
            "question": question,
            "answer": response["response"],
            "contexts": response["context"],
            "ground_truths": ground_truth,
        }

        dataset.append(data)

    with open("dataset/basic_rag_dataset1.json", "w") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)


def main():
    load_dotenv()
    print("Generating datasets...")
    generate_langchain_dataset()
    generate_haystack_dataset()
    generate_dspy_dataset()
    generate_llama_dataset()
    generate_crewai_dataset()
    generate_basic_rag_dataset()


if __name__ == "__main__":
    main()
