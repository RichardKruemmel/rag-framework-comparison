import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper as LangchainLLM
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils.utils import get_env_variable


def transform_dataset(path: str, context_is_string: bool = False):
    with open("dataset/{path}".format(path=path), "r") as file:
        json_data = json.load(file)

    new_dataset = {
        "question": [],
        "contexts": [],
        "answer": [],
        "ground_truth": [],
    }
    if context_is_string:
        for item in json_data:
            new_dataset["question"].append(item["question"])
            new_dataset["contexts"].append([item["contexts"]])
            new_dataset["answer"].append(item["answer"])
            new_dataset["ground_truth"].append(item["ground_truths"])
    else:
        for item in json_data:
            new_dataset["question"].append(item["question"])
            new_dataset["contexts"].append(item["contexts"])
            new_dataset["answer"].append(item["answer"])
            new_dataset["ground_truth"].append(item["ground_truths"])

    dataset = Dataset.from_dict(new_dataset)

    dataset.to_pandas()
    filename = "eval/eval_data/transformed_" + path
    with open(filename, "w") as file:
        json.dump(new_dataset, file, ensure_ascii=False, indent=4)
    return filename


def load_dataset(path):
    with open(path, "r") as file:
        json_data = json.load(file)
    dataset = Dataset.from_dict(json_data)
    return dataset


def setup_ragas_llm():
    load_dotenv()
    try:
        api_key = get_env_variable("OPENAI_API_KEY")
    except EnvironmentError as e:
        raise e

    llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4o")
    return LangchainLLM(llm)


def setup_ragas_embeddings():
    load_dotenv()
    try:
        openai_api_key = get_env_variable("OPENAI_API_KEY")
    except EnvironmentError as e:
        raise e

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=openai_api_key,
    )
    return embeddings


def run_ragas_evaluation(path: str):
    eval_llm = setup_ragas_llm()
    eval_embeddings = setup_ragas_embeddings()
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    faithfulness.llm = eval_llm
    faithfulness.embeddings = eval_embeddings
    answer_relevancy.llm = eval_llm
    context_precision.llm = eval_llm
    context_precision.embeddings = eval_embeddings
    context_recall.llm = eval_llm
    context_recall.embeddings = eval_embeddings
    dataset = load_dataset(path)
    result = evaluate(
        dataset=dataset, metrics=metrics, llm=eval_llm, embeddings=eval_embeddings
    )
    total_filename = path.replace("eval/eval_data/transformed", "results/total_results")
    with open(total_filename, "w") as file:
        json.dump(result, file, ensure_ascii=False, indent=4)
    filename = path.rsplit(".", 1)[0] + ".csv"
    results_filename = filename.replace("eval/eval_data/transformed", "results/results")
    csv = result.to_pandas()
    csv.to_csv(results_filename)


def transform_and_run_evaluation():
    datasets = [
        "basic_rag_dataset.json",
        "crewai_dataset.json",
        "dspy_dataset.json",
        "haystack_dataset.json",
        "llama_index_dataset.json",
        "langchain_dataset.json",
    ]

    for dataset in datasets:
        path = transform_dataset(dataset)
        run_ragas_evaluation(path)
