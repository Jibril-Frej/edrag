import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer


def embed(config: dict):
    """Embed the text chunks from the index using SentenceTransformer and save
    the embeddings in a csv file.

    :param config: configuration dictionary
    :type config: dic
    """
    # Read the index
    with open(config["IndexFile"], "r") as f:
        index = json.load(f)

    # Load the embedding model and tokenizer
    model = SentenceTransformer(config["ModelName"])

    # Get the text chunks in the correct order in a list
    chunks = [index[str(i)]["text"] for i in range(len(index))]

    # Embed the chunks
    embeddings = model.encode(chunks, batch_size=config["BatchSize"])

    # Save the embeddings
    np.savetxt(config["EmbeddingFile"], embeddings, delimiter=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.json")
    args = parser.parse_args()
    config = args.config

    with open(config, "r") as f:
        config = json.load(f)

    config = config["Embed"]

    embed(config)
