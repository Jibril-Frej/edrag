import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from omegaconf import DictConfig


log = logging.getLogger(__name__)


def retrieve(config: DictConfig, queries: list) -> np.ndarray:
    """Retrieve the most similar documents for each query.

    :param config: configuration dictionary
    :type config: DictConfig
    :param queries: list of queries to search for
    :type queries: list
    :return: indices of the most similar documents
    :rtype: np.ndarray
    """

    # Load the embedding model and tokenizer
    model = SentenceTransformer(config.Retrieval.Model)

    # Load and normalize the documents embeddings
    documents_embeddings = np.loadtxt(config.EmbeddingFile, delimiter=",")
    dnorm = np.linalg.norm(documents_embeddings, axis=1)
    documents_embeddings = documents_embeddings / dnorm[:, np.newaxis]

    log.info(
        f"Document Embeddings of shape {documents_embeddings.shape} loaded"
    )

    # Embed and normalize the queries
    queries_embeddings = model.encode(queries)
    qnorm = np.linalg.norm(queries_embeddings, axis=1)
    queries_embeddings = queries_embeddings / qnorm[:, np.newaxis]

    log.info(
        f"Queries Embeddings of shape {queries_embeddings.shape} computed"
    )

    # Calculate the cosine similarity between the queries and the embeddings
    similarities = np.dot(queries_embeddings, documents_embeddings.T)

    # Get the most similar documents for each query
    top_k = np.argsort(similarities, axis=1)[:, -config.Retrieval.TopK :][
        :, ::-1
    ]

    log.info(f"Top {config.Retrieval.TopK} documents retrieved for each query")

    return top_k
