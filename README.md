# RAG Framework Comparison

This project aims to compare various Retrieval-Augmented Generation (RAG) frameworks by evaluating different metrics. The frameworks included in this comparison are:

- LlamaIndex
- Haystack
- Dspy
- Langchain
- CrewAI
- Basic RAG

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Prerequisites

- Install Python 3.11 or higher
- Install [poetry](https://python-poetry.org/docs#installing-with-the-official-installer)

### Clone the Project

Clone the project repository from GitHub:

```bash
git clone https://github.com/RichardKruemmel/rag-framework-comparison.git
```

### Set Environment Variables

A `.sample.env` file is included in the project repository as a template for your own `.env` file. Copy the `.sample.env` file and rename the copy to `.env`:

```bash
cp .sample.env .env
```

## Installation

To install the required dependencies, use the following command:

```bash
# Create a virtual environment
poetry shell
# Install all packages
poetry install
```

Ensure you have Python 3.11 or higher installed.

## Usage

### Generating Metrics Overview

To run the experimental pipeline across different frameworks, run:

```bash
poetry run python3 -m main
```

To run parts of the experimental pipeline choose the sub directory of the feature you want to run (e.g. the dataset generation of each framework):

```bash
poetry run python3 -m dataset.main
```

## Project Structure

- `charts/`: Contains scripts for generating visualizations.
- `dataset/`: Contains datasets used for evaluation.
- `eval/`: Contains evaluation scripts and constants.
- `frameworks/`: Contains implementations of different RAG frameworks.
- `ingestion/`: Contains the script that ingested the election program pdf into the vector database
- `performance/`: Contains scripts and data for performance evaluation.
- `rag_from_scratch/`: Contains a basic implementation of RAG from scratch.
- `results/`: Contains the total results and more detailed results of each framework
- `utils/`: Contains utility functions.
- `vector_db/`: Contains functions required for the vector database

### Key Files

- `main.py`: Main script for executing the pipeline: it generates the datasets, evaluates the performance and the evaluates the quality.
- `dataset/main.py`: Main Script for generating datasets for each framework.
- `performance/main.py`: Main script for performance evaluation.
- `eval/main.py`: Script for quality evaluation of datasets.
- `charts/main.py`: Main script for generating charts based on the evaluation.
