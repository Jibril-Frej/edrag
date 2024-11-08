import os
import json
import logging

from omegaconf import DictConfig
from hydra.utils import get_original_cwd

log = logging.getLogger(__name__)


def update_index(
    config: DictConfig,
    index: dict,
    doc_name: str,
    doc_text: str,
):
    """Reads a document and updates the index with the chunks of the document.

    :param config: configuration dictionary
    :type config: DictConfig
    :param index: dictionary to store the index.
    :type index: dict
    :param doc_name: name of the document
    :type doc_name: str
    :param doc_text: text of the document
    :type doc_text: str
    """
    chunk_size = config.ChunkSize
    chunk_overlap = config.ChunkOverlap

    chunk_index = 0

    # Loop through all the characters in the document
    for i in range(0, len(doc_text), chunk_size - chunk_overlap):

        # Get the chunk
        chunk = doc_text[i : i + chunk_size]

        # Update the index
        index[len(index)] = {
            "file_name": doc_name,
            "chunk_index": chunk_index,
            "start_chr": i,
            "end_chr": i + chunk_size,
            "text": chunk,
        }

        chunk_index += 1


def basic_indexing(config: DictConfig):
    """Indexes all the documents in the DocumentsDirectory and saves the index
    in the IndexFile.

    :param config: configuration dictionary
    :type config: DictConfig
    """

    index = dict()

    # Get the list of documents
    docdir = os.path.join(get_original_cwd(), config.DocumentsDirectory)

    docs = os.listdir(docdir)

    log.info(f"Indexing {len(docs)} documents from {docdir}")

    # Loop through all the documents
    for doc_name in docs:
        doc_path = os.path.join(docdir, doc_name)

        # Read the document
        with open(doc_path, "r") as f:
            doc_text = f.read()

        # Update the index
        update_index(config, index, doc_name, doc_text)

    log.info(f"Indexed {len(index)} chunks")

    # Save the index
    log.info(f"Saving the index to {config.IndexFile}")

    with open(config.IndexFile, "w") as f:
        json.dump(index, f)
