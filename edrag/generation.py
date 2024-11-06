import json
import argparse
import yaml

import numpy as np
from openai import OpenAI

from retrieval import retrieve, retrieve_all


def make_messages(config: dict, query: str, documents: list) -> list:
    """Generate a prompt for the generation model.

    :param config: configuration dictionary
    :type config: dict
    :param query: query to generate a response to
    :type query: str
    :param documents: list of documents
    :type documents: list
    :return: generated prompt
    :rtype: list
    """

    # Read the system prompt from the configuration
    with open(config["PromptPath"], "r") as f:
        system_prompt = yaml.safe_load(f)

    messages = [system_prompt]

    # Add the query and documents to the prompt
    content = f"<query>\n{query}\n</query>\n\n"

    for doc in documents:
        content += f"<document>\n{doc["text"]}\n</document>\n\n"

    messages.append({"role": "user", "content": content})

    return messages


def generate(config: dict, query: str) -> str | None:
    """Generate a response to a query using the retrieval module to find the
    most similar documents and then using the generation model to generate a
    response.

    :param config: configuration dictionary
    :type config: dict
    :param query: query to generate a response to
    :type query: str
    :return: generated response
    :rtype: str
    """

    # Retrieve the indices of the most similar documents
    top_k = retrieve(config["Retrieval"], query)

    config = config["Generation"]

    # Load the documents
    with open(config["IndexFile"], "r") as f:
        index = json.load(f)

    # Get the most similar documents
    documents = [index[str(i)] for i in top_k]

    # Generate the messages
    messages = make_messages(config, query, documents)

    # Generate the response
    client = OpenAI()

    completion = client.chat.completions.create(
        model=config["ModelName"],
        messages=messages,
        temperature=config["Temperature"],
    )

    response = completion.choices[0].message.content

    return response


def generate_all(config: dict, queries: list) -> tuple[list, np.ndarray]:
    """Generate responses to a list of queries.

    :param config: configuration dictionary
    :type config: dict
    :param queries: list of queries to generate responses to
    :type queries: list
    :return: list of generated responses
    :rtype: list
    """

    # Retrieve the indices of the most similar documents for each query
    top_k = retrieve_all(config["Retrieval"], queries)

    config = config["Generation"]

    # Load the documents
    with open(config["IndexFile"], "r") as f:
        index = json.load(f)

    responses = []

    for query, top_k_documents in zip(queries, top_k):
        # Get the most similar documents
        documents = [index[str(i)] for i in top_k_documents]

        # Generate the messages
        messages = make_messages(config, query, documents)

        # Generate the response
        client = OpenAI()

        completion = client.chat.completions.create(
            model=config["ModelName"],
            messages=messages,
            temperature=config["Temperature"],
        )

        response = completion.choices[0].message.content

        responses.append(response)

    return responses, top_k


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

    query = """
    Given two posets, ($S, \\preceq_{1}$) and $(T, \\preceq_{2})$, show that
    ($S \\times T$, $\\preceq$) forms a poset when ordered by the relation
    $\\preceq$ defined as: $(s,t) \\preceq (u,v)$ if and only if $s
    \\preceq_1 u$ and $t \\preceq_2 v$.
    """

    response = generate(config, query)
