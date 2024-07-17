import logging
from ingestion.metadata import ELECTION_ID, PARTY_ID, PROGRAM_ID
from llama_index import SimpleDirectoryReader

programs = {
    "SPD": "https://www.abgeordnetenwatch.de/sites/default/files/election-program-files/SPD_Wahlprogramm_BTW2021.pdf",
    "FDP": "https://www.abgeordnetenwatch.de/sites/default/files/election-program-files/FDP_Wahlprogramm_BTW2021.pdf",
    "AFD": "https://www.abgeordnetenwatch.de/sites/default/files/election-program-files/AfD_Wahlprogramm_BTW2021.pdf",
    "CDU": "https://www.abgeordnetenwatch.de/sites/default/files/election-program-files/CDU-CSU_Wahlprogramm_BTW2021.pdf",
    "Grüne": "https://www.abgeordnetenwatch.de/sites/default/files/election-program-files/B90DieGr%C3%BCnen_Wahlprogramm_BTW2021.pdf",
}


def load_docs(save_path: str):
    documents = SimpleDirectoryReader(input_files=[save_path]).load_data()
    logging.info(f"Loaded PDF into Document object")

    for doc in documents:
        metadata = doc.metadata
        extra_info = get_metadata()
        metadata.update(extra_info)
        doc.metadata = metadata
    return documents


def get_metadata():
    return {
        "election_id": ELECTION_ID,
        "party_id": PARTY_ID,
        "election_program_id": PROGRAM_ID,
        "group_id": PROGRAM_ID,
    }
