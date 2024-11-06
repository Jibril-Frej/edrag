import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer


def retrieve(config: dict, query: str) -> np.ndarray:
    """Retrieve the most similar documents to the query.

    :param config: configuration dictionary
    :type config: dict
    :param query: query to search for
    :type query: str
    :return: indices of the most similar documents
    :rtype: np.ndarray
    """

    # Load the embedding model and tokenizer
    model = SentenceTransformer(config["ModelName"])

    # Load the embeddings
    embeddings = np.loadtxt(config["EmbeddingFile"], delimiter=",")

    # Embed the query
    query_embedding = model.encode(query)

    # Calculate the cosine similarity between the query and the embeddings
    similarities = np.dot(embeddings, query_embedding.T) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    # Get the most similar documents
    top_k = np.argsort(similarities, axis=0)[-config["TopK"] :][::-1]

    return top_k


def retrieve_all(config: dict, queries: list) -> np.ndarray:
    """Retrieve the most similar documents for each query.

    :param config: configuration dictionary
    :type config: dict
    :param queries: list of queries to search for
    :type queries: list
    :return: indices of the most similar documents
    :rtype: np.ndarray
    """

    # Load the embedding model and tokenizer
    model = SentenceTransformer(config["ModelName"])

    # Load and normalize the documents embeddings
    documents_embeddings = np.loadtxt(config["EmbeddingFile"], delimiter=",")
    dnorm = np.linalg.norm(documents_embeddings, axis=1)
    documents_embeddings = documents_embeddings / dnorm[:, np.newaxis]

    # Embed and normalize the queries
    queries_embeddings = model.encode(queries)
    qnorm = np.linalg.norm(queries_embeddings, axis=1)
    queries_embeddings = queries_embeddings / qnorm[:, np.newaxis]

    # Calculate the cosine similarity between the queries and the embeddings
    similarities = np.dot(queries_embeddings, documents_embeddings.T)

    # Get the most similar documents for each query
    top_k = np.argsort(similarities, axis=1)[:, -config["TopK"] :][:, ::-1]

    return top_k


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/AICC_2023.json",
    )  # type: ignore
    args = parser.parse_args()
    config = args.config

    with open(config, "r") as f:
        config = json.load(f)

    config = config["Retrieval"]

    queries = [
        "\n\n\\noindent Advanced information, computation, communication I\\\\\nEPFL - Fall semester 2023-2024\\\\\n\\setcounter{exno}{0}\n\n\\begin{center}{\\huge Week 3 --- Solutions}\\\\\nOctober 6, 2023\n\\end{center}\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n \n\n\\problem\n\\emph{Suppose you want to prove that every product of integers of the form $k (k+1)(k+2)$ is divisible by 6.\nIf you want to prove this by cases, which of the following is a set of cases you should use?}\n\\begin{description}\n    \\item[$\\qquad\\bigcirc$] the product ends in 3; the product ends in 6; the product ends in 9.\n    \\item[$\\qquad\\,\\checkmark$] when k is divided by 3, the remainder is 0; when $k$ is divided by 3, the remainder is 1; when k is divided by 3, the remainder is 2.\n    \\item[$\\qquad\\bigcirc$] $k = 3^n ; k \\neq 3^n$.\n    \\item[$\\qquad\\bigcirc$] $k$ is prime, $k$ is not prime.\n\\end{description}\n\nThese three cases allow you to conclude that one of the three numbers in $k (k + 1)(k + 2)$ is a multiple of 3.\nIf the remainder when $k$ is divided by 3 is 0, the number $k$ is a m",
        " $y$, one of the two numbers is less than the other.\nIf you assume $x < y$, you can write $x + x < x + y < y + y$, or $2x < x + y < 2y$.\nDivide by 2 to obtain $x < \\frac{x+y}{2} < y$.\nIf $y < x$, the first argument can be used, interchanging the roles of $x$ and $y$ : $y + y < y + x < x + x$, or $2y < y + x < 2x$.\nDivide by 2 to obtain $y < \\frac{x+y}{2} < x$.\n\n\\problem\n\\emph{Suppose you want to prove a theorem about the product of absolute values of real numbers, $|x| . |y|$.\nIf you were to give a proof by cases, what set of cases would probably be the best to use?}\n\\begin{description}\n    \\item[$\\qquad\\,\\checkmark$] both $x$ and $y$ nonnegative; one negative and one nonnegative; both negative.\n    \\item[$\\qquad\\bigcirc$] both $x$ and $y$ rational; one rational and one irrational; both irrational.\n    \\item[$\\qquad\\bigcirc$] both $x$ and $y$ even; one even and one odd; both odd.\n    \\item[$\\qquad\\bigcirc$] $x > y$; $x < y$; $x = y$.\n\\end{description}\nThe definition of absolute value $|x|$ depends on two case",
    ]

    top_k = retrieve_all(config, queries)

    print(top_k)
