import argparse
import json
import yaml

from openai import OpenAI

from generation import generate_all


def evaluate_answer(config: dict, solution: str, answer: str) -> str | None:
    """Evaluate the answer.

    :param config: configuration dictionary
    :type config: dict
    :param solution: correct answer
    :type solution: str
    :param answer: generated answer
    :type answer: str
    :return: evaluation result
    :rtype: str
    """

    # Read the evaluation prompt from the configuration
    with open(config["PromptPath"], "r") as f:
        evaluation_prompt = yaml.safe_load(f)

    messages = [evaluation_prompt]

    # Add the solution and answer to the prompt
    content = f"<question_solution>\n{answer}\n</question_solution>\n\n"
    content += f"<answer>\n{solution}\n</answer>\n\n"

    messages.append({"role": "user", "content": content})

    client = OpenAI()

    # Generate the evaluation
    completion = client.chat.completions.create(
        model=config["ModelName"],
        messages=messages,
        temperature=config["Temperature"],
    )

    response = completion.choices[0].message.content

    return response


def evaluate_all(config: dict) -> None:
    """Evaluate the RAG system.

    :param config: configuration dictionary
    :type config: dict
    """

    eval_config = config["Evaluation"]

    # Load the queries
    with open(eval_config["QSFile"], "r") as f:
        questions_solutions = json.load(f)

    # Get only the fist k queries
    k = 10
    questions_solutions = {i: questions_solutions[str(i)] for i in range(k)}

    all_questions = [questions_solutions[i]["question"] for i in range(k)]
    all_answers, top_k = generate_all(config, all_questions)

    results = {}

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
            label = evaluate_answer(eval_config, solution, answer)
        results[qs_id]["label"] = label

    with open(eval_config["ResultsFile"], "w") as f:
        json.dump(results, f, indent=4)


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

    evaluate_all(config)
