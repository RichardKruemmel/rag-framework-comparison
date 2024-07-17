import os
import json


def get_env_variable(var_name):
    value = os.getenv(var_name)
    if value is None:
        raise EnvironmentError(f"{var_name} not found in environment variables")
    return value


def save_dataset_to_json(dataset, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset.dict(), f, ensure_ascii=False, indent=4)


def match_crewdataset_to_context():
    with open("dataset/crewai_dataset_without_context.json", "r") as file:
        dataset = json.load(file)

    with open("dataset/crew_ai_context.json", "r") as file:
        context_map = json.load(file)
    context_index = 0
    for data in dataset:
        if data["contexts"] != [""]:
            data["contexts"] = context_map[context_index]
            context_index += 1

    with open("dataset/crewai_dataset.json", "w") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)
