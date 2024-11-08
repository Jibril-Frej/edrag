import json
import logging

import numpy as np
from openai import OpenAI
from omegaconf import DictConfig

from retrieval import retrieve

log = logging.getLogger(__name__)


def make_generation_messages(
    config: DictConfig, query: str, documents: list
) -> list:
    """Generate a prompt for the generation model.

    :param config: configuration dictionary
    :type config: DictConfig
    :param query: query to generate a response to
    :type query: str
    :param documents: list of documents
    :type documents: list
    :return: generated prompt
    :rtype: list
    """

    # Read the system prompt from the configuration
    system_prompt = {"role": "system", "content": config.Generation.Prompt}

    messages = [system_prompt]

    # Add the query and documents to the prompt
    content = f"<query>\n{query}\n</query>\n\n"

    for doc in documents:
        content += f"<document>\n{doc["text"]}\n</document>\n\n"

    messages.append({"role": "user", "content": content})

    return messages


def generate(config: DictConfig, queries: list) -> tuple[list, np.ndarray]:
    """Generate responses to a list of queries.

    :param config: configuration dictionary
    :type config: DictConfig
    :param queries: list of queries to generate responses to
    :type queries: list
    :return: list of generated responses
    :rtype: list
    """

    # Retrieve the indices of the most similar documents for each query
    top_k = retrieve(config, queries)

    # Load the index
    with open(config.IndexFile, "r") as f:
        index = json.load(f)

    client = OpenAI()
    responses = []

    log.info(f"Generating responses to {len(queries)} queries")
    for query, top_k_documents in zip(queries, top_k):
        # Get the most similar documents
        documents = [index[str(i)] for i in top_k_documents]

        log.info(f"Generating response: {len(responses) + 1}/{len(queries)}")

        # Generate the messages
        messages = make_generation_messages(config, query, documents)

        completion = client.chat.completions.create(
            model=config.Generation.Model,
            messages=messages,
            temperature=config.Generation.Temperature,
        )

        response = completion.choices[0].message.content

        responses.append(response)

    return responses, top_k
