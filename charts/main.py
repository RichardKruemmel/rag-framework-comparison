import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json


def create_heatmap(path: str, name: str):
    df = pd.read_csv(path)
    normalized_df = df[
        ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    ]

    plt.figure(figsize=(12, 24))
    heatmap = sns.heatmap(
        normalized_df,
        cmap="RdYlGn",
        cbar=True,
        linewidths=0.5,
    )
    plt.xticks(rotation=0)
    plt.yticks(ticks=range(len(normalized_df)), labels=normalized_df.index, rotation=0)
    plt.title("Heatmap of Metrics for Each Question of {name}".format(name=name))
    plt.xlabel("Metrics")
    heatmap.set_ylabel("Question Index", fontsize=24, labelpad=10)

    file = path.replace(".csv", "")
    plt.savefig("charts/heatmap_{file}.png".format(file=file))


def create_radar_chart(path: str, name: str):
    with open(path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data, index=[0])
    labels = list(df.columns)
    values = df.values.flatten().tolist()
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_ylim(0, 1)

    ax.plot(angles, values, color="green", linewidth=2, linestyle="solid")
    ax.fill(angles, values, color="green", alpha=0.25)

    plt.title(
        "Radar Chart of Metrics for {name}".format(name=name),
        size=20,
        color="green",
        y=1.1,
    )
    file = path.replace(".json", "").replace("total_results_", "")
    plt.savefig("charts/radar_chart_{file}.png".format(file=file))


def generate_metric_overview():
    file_path = "results/total_results_all.json"
    with open(file_path, "r") as file:
        data = json.load(file)

    metrics = list(data[next(iter(data))].keys())
    frameworks = list(data.keys())

    for metric in metrics:
        metric_values = [data[framework][metric] for framework in frameworks]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            frameworks, metric_values, color=plt.cm.Paired(range(len(frameworks)))
        )

        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 0.01,
                round(yval, 2),
                ha="center",
                va="bottom",
            )

        plt.ylim(0, 1)
        plt.ylabel(metric.capitalize().replace("_", " "))
        plt.title(
            f"{metric.capitalize().replace('_', ' ')} Across Frameworks",
            fontweight="bold",
        )
        plt.savefig(f"charts/{metric}_overview.png")


def generate_average_setup_times():

    file_path = "performance/average_setup_times.json"
    with open(file_path, "r") as file:
        setup_times = json.load(file)

    setup_times_ms = {framework: time * 1000 for framework, time in setup_times.items()}

    frameworks = list(setup_times_ms.keys())
    times_ms = list(setup_times_ms.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(frameworks, times_ms, color=plt.cm.Paired(range(len(frameworks))))

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            round(yval, 2),
            ha="center",
            va="bottom",
        )

    plt.ylabel("Average Setup Time (ms)")
    plt.title("Average Setup Time for Each Framework (n=30)", fontweight="bold")
    plt.savefig("charts/average_setup_times_overview.png")


def generate_average_response_times():
    file_path = "performance/avg_answer_time.json"
    with open(file_path, "r") as file:
        response_times = json.load(file)

    frameworks = list(response_times.keys())
    times = list(response_times.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(frameworks, times, color=plt.cm.Paired(range(len(frameworks))))

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            round(yval, 2),
            ha="center",
            va="bottom",
        )

    plt.ylabel("Average Response Time (s)")
    plt.title("Average Response Time for Each Framework (n=30)", fontweight="bold")
    plt.figtext(
        0.25,
        0.02,
        "Note: Times measured for the first 10 questions from the golden dataset",
    )
    plt.savefig("charts/average_response_times_overview.png")


def calculate_variance(values):
    mean = sum(values) / len(values)
    squared_diffs = [(x - mean) ** 2 for x in values]
    variance = sum(squared_diffs) / len(values)
    return variance


def generate_variance_overview():
    frameworks = ["llama_index", "haystack", "dspy"]
    metrics = [
        "Faithfulness",
        "Answer Relevancy",
        "Context Precision",
        "Context Recall",
    ]
    variances = {"Framework": [], "Metric": [], "Variance": []}

    framework_colors = {
        "LlamaIndex": plt.cm.Paired(1),
        "Haystack": plt.cm.Paired(2),
        "Dspy": plt.cm.Paired(3),
    }

    for framework in frameworks:
        try:
            with open(f"results/total_results_{framework}_dataset.json", "r") as file:
                total_results = json.load(file)
            with open(
                f"results/variance_total_results_{framework}_dataset.json", "r"
            ) as file:
                variance_results = json.load(file)

            for metric in metrics:
                total_value = total_results[metric.lower().replace(" ", "_")]
                variance_value = variance_results[metric.lower().replace(" ", "_")]
                var = calculate_variance([total_value, variance_value])
                framework_name = (
                    framework.capitalize()
                    if framework != "llama_index"
                    else "LlamaIndex"
                )
                variances["Framework"].append(framework_name)
                variances["Metric"].append(metric)
                variances["Variance"].append(var)
        except FileNotFoundError:
            print(f"Data files for {framework} not found.")

    df = pd.DataFrame(variances)
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(metrics))
    width = 0.2

    for i, framework in enumerate(frameworks):
        framework_name = (
            framework.capitalize() if framework != "llama_index" else "LlamaIndex"
        )
        subset = df[df["Framework"] == framework_name]
        positions = x + i * width
        bars = ax.bar(
            positions,
            subset["Variance"],
            width,
            label=framework_name,
            color=framework_colors[framework_name],
        )

        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                f"{yval:.2e}" if yval != 0.0 else "0.0",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_title("Variance Comparison Across Metrics and Frameworks", fontweight="bold")
    ax.set_ylabel("Variance")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.legend(title="Framework", bbox_to_anchor=(0.45, 1), loc="upper left")
    plt.savefig("charts/variance_comparison.png")


def generate_all_heatmaps():
    create_heatmap(path="results/results_crewai_dataset.csv", name="Crew AI Dataset")
    create_heatmap(
        path="results/results_llama_index_dataset.csv", name="LLAMA Index Dataset"
    )
    create_heatmap(path="results/results_dspy_dataset.csv", name="DSpy Dataset")
    create_heatmap(path="results/results_haystack_dataset.csv", name="Haystack Dataset")
    create_heatmap(
        path="results/results_langchain_dataset.csv", name="Langchain Dataset"
    )
    create_heatmap(
        path="results/results_basic_rag_dataset.csv", name="Basic RAG Dataset"
    )


def generate_all_radar_charts():
    create_radar_chart(
        path="results/total_results_crewai_dataset.json", name="Crew AI Dataset"
    )
    create_radar_chart(
        path="results/total_results_llama_index_dataset.json",
        name="LLAMA Index Dataset",
    )
    create_radar_chart(
        path="results/total_results_dspy_dataset.json", name="DSpy Dataset"
    )
    create_radar_chart(
        path="results/total_results_haystack_dataset.json", name="Haystack Dataset"
    )
    create_radar_chart(
        path="results/total_results_langchain_dataset.json", name="Langchain Dataset"
    )
    create_radar_chart(
        path="results/total_results_basic_rag_dataset.json", name="Basic RAG Dataset"
    )


def create_radar_chart_with_multiple_datasets(
    paths: list, names: list, output_file: str
):
    colors = ["green", "blue", "red", "purple", "orange"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, (path, name) in enumerate(zip(paths, names)):
        with open(path, "r") as f:
            data = json.load(f)

        df = pd.DataFrame(data, index=[0])
        labels = list(df.columns)
        values = df.values.flatten().tolist()
        values += values[:1]

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        ax.plot(
            angles,
            values,
            color=colors[i % len(colors)],
            linewidth=2,
            linestyle="solid",
            label=name,
        )
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_ylim(0, 1)

    plt.title("Radar Chart of All Datasets", size=20, color="green", y=1.1)
    plt.legend(loc="upper right")
    plt.savefig(output_file)


def generate_multiple_datasets_radar_chart():
    paths = [
        "results/total_results_crewai_dataset.json",
        "results/total_results_llama_index_dataset.json",
        "results/total_results_dspy_dataset.json",
        "results/total_results_haystack_dataset.json",
        "results/total_results_langchain_dataset.json",
        "results/total_results_basic_rag_dataset.json",
    ]
    names = [
        "Crew AI Dataset",
        "LLAMA Index Dataset",
        "DSpy Dataset",
        "Haystack Dataset",
        "Langchain Dataset",
        "Basic RAG Dataset",
    ]
    create_radar_chart_with_multiple_datasets(
        paths, names, "charts/radar_chart_multiple_datasets.png"
    )


def main():
    generate_metric_overview()
    generate_average_setup_times()
    generate_average_response_times()
    generate_variance_overview()


if __name__ == "__main__":
    main()
