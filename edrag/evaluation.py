import os
import json
import logging

from openai import OpenAI
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

from generation import generate


log = logging.getLogger(__name__)


def make_evaluation_messages(
    config: DictConfig, solution: str, answer: str
) -> list:

    # Read the evaluation prompt from the configuration
    evaluation_prompt = {"role": "system", "content": config.Evaluation.Prompt}

    messages = [evaluation_prompt]

    # Add the solution and answer to the prompt
    content = f"<question_solution>\n{answer}\n</question_solution>\n\n"
    content += f"<answer>\n{solution}\n</answer>\n\n"

    messages.append({"role": "user", "content": content})

    return messages


def evaluate_answer(
    config: DictConfig, solution: str, answer: str
) -> str | None:
    """Evaluate the answer.

    :param config: configuration dictionary
    :type config: DictConfig
    :param solution: correct answer
    :type solution: str
    :param answer: generated answer
    :type answer: str
    :return: evaluation result
    :rtype: str
    """

    # Generate the evaluation prompt
    messages = make_evaluation_messages(config, solution, answer)

    client = OpenAI()

    # Generate the evaluation
    completion = client.chat.completions.create(
        model=config.Evaluation.Model,
        messages=messages,
        temperature=config.Evaluation.Temperature,
    )

    response = completion.choices[0].message.content

    return response


def evaluate_all(config: DictConfig) -> None:
    """Evaluate the RAG system.

    :param config: configuration dictionary
    :type config: DictConfig
    """

    # Load the queries
    query_path = os.path.join(get_original_cwd(), config.QSFile)
    with open(query_path, "r") as f:
        questions_solutions = json.load(f)

    log.info(
        f"Loaded {len(questions_solutions)} questions and solutions"
        f"from {query_path}"
    )

    # Get only the fist k queries
    k = 1
    questions_solutions = {i: questions_solutions[str(i)] for i in range(k)}

    all_questions = [questions_solutions[i]["question"] for i in range(k)]
    all_answers, top_k = generate(config, all_questions)

    results = {}

    log.info(f"Generating answers to {len(all_questions)} questions")

    # Generate and evaluate the answers
    for qs_id, qs in questions_solutions.items():
        question = qs["question"]
        solution = qs["solution"]
        answer = all_answers[qs_id]
        retrieved_docs = top_k[qs_id].tolist()
        relevant_docs = qs["relevant_documents"]
        results[qs_id] = {
            "question": question,
            "solution": solution,
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "relevant_docs": relevant_docs,
        }
        if answer is None:
            label = "-1"
        else:
            label = evaluate_answer(config, solution, answer)
        results[qs_id]["label"] = label
        log.info(f"Question {qs_id} evaluated as {label}")

    with open(config.ResultsFile, "w") as f:
        json.dump(results, f, indent=4)

    log.info(f"Results saved to {config.ResultsFile}")
