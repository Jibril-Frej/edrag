import json
import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from omegaconf import DictConfig


log = logging.getLogger(__name__)


def embed(config: DictConfig):
    """Embed the text chunks from the index using SentenceTransformer and save
    the embeddings in a csv file.

    :param config: configuration dictionary
    :type config: DictConfig
    """

    # Read the index
    with open(config.IndexFile, "r") as f:
        index = json.load(f)

    # Load the embedding model and tokenizer
    model = SentenceTransformer(config.Embedding.Model)

    # Get the text chunks in the correct order in a list
    chunks = [index[str(i)]["text"] for i in range(len(index))]

    # Embed the chunks
    embeddings = model.encode(chunks, batch_size=config.Embedding.BatchSize)

    log.info(f"Embeddings of shape {embeddings.shape} computed")

    # Save the embeddings as a csv file with savetxt for readability
    np.savetxt(config.EmbeddingFile, embeddings, delimiter=",")

    log.info(f"Embeddings saved to {config.EmbeddingFile}")
