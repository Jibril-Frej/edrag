import os
import argparse
import json


def update_index(
    config: dict,
    index: dict,
    doc_name: str,
    doc_text: str,
):
    """Reads a document and updates the index with the chunks of the document.

    :param config: configuration dictionary
    :type config: dict
    :param index: dictionary to store the index. Keys are doc_name with chunk index appended and values are the chunk text
    :type index: dict
    :param doc_name: name of the document
    :type doc_name: str
    :param doc_text: text of the document
    :type doc_text: str
    """
    chunk_size = config["ChunkSize"]
    chunk_overlap = config["ChunkOverlap"]

    chunk_index = 0

    # Loop through all the characters in the document
    for i in range(0, len(doc_text), chunk_size - chunk_overlap):

        # Get the chunk
        chunk = doc_text[i : i + chunk_size]

        # Update the index
        chunk_id = f"{doc_name}_{chunk_index}"
        index[chunk_id] = chunk
        chunk_index += 1


def basic_indexing(config: dict):
    """Indexes all the documents in the DocumentsDirectory and saves the index in the IndexFile.

    :param config: configuration dictionary
    :type config: dict
    """
    index = dict()

    # Get the list of documents
    docs = os.listdir(config["DocumentsDirectory"])

    # Loop through all the documents
    for doc_name in docs:
        doc_path = os.path.join(config["DocumentsDirectory"], doc_name)

        # Read the document
        with open(doc_path, "r") as f:
            doc_text = f.read()

        # Update the index
        update_index(config, index, doc_name, doc_text)

    # Save the index
    with open(config["IndexFile"], "w") as f:
        json.dump(index, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/indexing.json")
    args = parser.parse_args()
    config = args.config

    with open(config, "r") as f:
        config = json.load(f)

    basic_indexing(config)
