import json
import time

from dotenv import load_dotenv

from frameworks.crewai import setup_crew, setup_perf_crew
from frameworks.dspy import setup_dspy
from frameworks.haystack import setup_haystack
from frameworks.langchain import setup_perf_langchain
from frameworks.llama_index import setup_llama_index
from rag_from_scratch.main import Basic_RAG


def performance_test():
    setup_times = {
        "Langchain": [],
        "Haystack": [],
        "Dspy": [],
        "LlamaIndex": [],
        "CrewAI": [],
        "Basic_RAG": [],
    }
    n = 30  # Number of times to run the setup process for each framework and the answer time test
    qa_pair_count = 10  # Number of QA pairs to use for the answer time test
    for _ in range(n):
        start_time = time.time()
        setup_perf_langchain()
        setup_times["Langchain"].append(time.time() - start_time)

        start_time = time.time()
        setup_haystack()
        setup_times["Haystack"].append(time.time() - start_time)

        start_time = time.time()
        setup_dspy()
        setup_times["Dspy"].append(time.time() - start_time)

        start_time = time.time()
        setup_llama_index()
        setup_times["LlamaIndex"].append(time.time() - start_time)

        start_time = time.time()
        setup_crew()
        setup_times["CrewAI"].append(time.time() - start_time)

        start_time = time.time()
        Basic_RAG()
        setup_times["Basic_RAG"].append(time.time() - start_time)

    avg_setup_times = {
        framework: sum(times) / n for framework, times in setup_times.items()
    }

    with open("performance/individual_setup_times.json", "w") as file:
        json.dump(setup_times, file, indent=4)

    with open("performance/average_setup_times.json", "w") as file:
        json.dump(avg_setup_times, file, indent=4)

    with open("eval/updated_qa_pairs.json", "r") as file:
        qa_pairs = json.load(file)

    answer_times = {
        "Langchain": [],
        "Haystack": [],
        "Dspy": [],
        "LlamaIndex": [],
        "CrewAI": [],
        "Basic_RAG": [],
    }
    langchain = setup_perf_langchain()

    for i in range(n):
        start = time.time()
        for pair in qa_pairs[:qa_pair_count]:
            langchain.invoke(pair["question"])
        end = time.time()
        answer_times["Langchain"].append(end - start)

    haystack = setup_haystack()
    for i in range(n):
        start = time.time()
        for pair in qa_pairs[:qa_pair_count]:
            haystack.run(
                {
                    "text_embedder": {"text": pair["question"]},
                    "prompt_builder": {"question": pair["question"]},
                    "answer_builder": {"query": pair["question"]},
                }
            )
        end = time.time()
        answer_times["Haystack"].append(end - start)

    dspy = setup_dspy()
    for i in range(n):
        start = time.time()
        for pair in qa_pairs[:qa_pair_count]:
            dspy(pair["question"])
        end = time.time()
        answer_times["Dspy"].append(end - start)

    llama = setup_llama_index()
    for i in range(n):
        start = time.time()
        for pair in qa_pairs[:qa_pair_count]:
            llama.query(pair["question"])
        end = time.time()
        answer_times["LlamaIndex"].append(end - start)

    crew = setup_perf_crew()
    for i in range(n):
        start = time.time()
        for pair in qa_pairs[:qa_pair_count]:
            crew.kickoff(inputs={"question": pair["question"]})
        end = time.time()
        answer_times["CrewAI"].append(end - start)

    basic_rag = Basic_RAG()
    for i in range(n):
        start = time.time()
        for pair in qa_pairs[:qa_pair_count]:
            basic_rag.query(pair["question"])
        end = time.time()
        answer_times["Basic_RAG"].append(end - start)

    average_answer_time = {
        framework: sum(times) / n for framework, times in answer_times.items()
    }

    with open("performance/answer_times.json", "w") as file:
        json.dump(answer_times, file, ensure_ascii=False, indent=4)
    with open("performance/avg_answer_time.json", "w") as file:
        json.dump(average_answer_time, file, ensure_ascii=False, indent=4)


def main():
    load_dotenv()
    print("Running performance test...")
    performance_test()


if __name__ == "__main__":
    main()
