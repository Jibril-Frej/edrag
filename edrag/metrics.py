import json
import logging

import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def compute_hit_rate(retrieved: list, relevant: list, k: int) -> float:
    """Compute the hit rate at k.

    :param retrieved: list of retrieved documents
    :type retrieved: list
    :param relevant: list of relevant documents
    :type relevant: list
    :param k: number of retrieved documents to consider
    :type k: int
    :return: hit rate at k
    :rtype: float
    """
    return len(set(retrieved[:k]) & set(relevant)) / k


def compute_ndcg(retrieved: list, relevant: list, k: int) -> float:
    """Compute the normalized discounted cumulative gain at k.

    :param retrieved: list of retrieved documents
    :type retrieved: list
    :param relevant: list of relevant documents
    :type relevant: list
    :param k: number of retrieved documents to consider
    :type k: int
    :return: normalized discounted cumulative gain at k
    :rtype: float
    """
    dcg = 0
    k = min(k, len(retrieved))
    for i in range(k):
        if retrieved[i] in relevant:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg


def compute_generation_metrics(results: dict) -> dict:
    """Compute the generation metrics.

    :param results: results dictionary
    :type results: dict
    :return: generation metrics
    :rtype: dict
    """
    generation_metrics = {
        "correct": 0.0,
        "undecided": 0.0,
        "incorrect": 0.0,
        "wrong_evaluator": 0.0,
    }

    for qs in results.values():
        label = qs["label"]
        if label == "1":
            generation_metrics["correct"] += 1
        elif label == "0":
            generation_metrics["undecided"] += 1
        elif label == "-1":
            generation_metrics["incorrect"] += 1
        else:
            generation_metrics["wrong_evaluator"] += 1

    for key, value in generation_metrics.items():
        generation_metrics[key] = value / len(results)

    return {"generation": generation_metrics}


def compute_retrieval_metrics(
    config: DictConfig, results: dict, metrics: dict
) -> dict:
    """Compute the retrieval metrics and update the metrics dictionary with the
    average and std of the retrieval metrics.

    :param config: configuration dictionary
    :type config: DictConfig
    :param results: results dictionary
    :type results: dict
    :param metrics: metrics dictionary
    :type metrics: dict
    :return: retrieval metrics
    :rtype: dict
    """
    retrieval_metrics = {}

    metrics["retrieval"] = {}
    k = config.Evaluation.K

    for qs_id, qs in results.items():
        retrieved_docs = qs["retrieved_docs"]
        relevant_docs = qs["relevant_docs"]
        hit_rate = compute_hit_rate(retrieved_docs, relevant_docs, k)
        ndcg = compute_ndcg(retrieved_docs, relevant_docs, k)
        retrieval_metrics[qs_id] = {"hit_rate": hit_rate, "ndcg": ndcg}

    for metric in ["hit_rate", "ndcg"]:
        avg = np.mean([v[metric] for v in retrieval_metrics.values()])
        std = np.std([v[metric] for v in retrieval_metrics.values()])
        metrics["retrieval"][f"avg_{metric}"] = avg
        metrics["retrieval"][f"std_{metric}"] = std

    return retrieval_metrics


def compute_all_metrics(config: DictConfig) -> None:
    """Compute all the metrics.

    :param config: configuration dictionary
    :type config: DictConfig
    """
    with open(config.ResultsFile, "r") as f:
        results = json.load(f)

    log.info(f"Loaded results from {config.ResultsFile}")

    metrics = compute_generation_metrics(results)

    log.info("Generation metrics computed")

    retrieval_metrics = compute_retrieval_metrics(config, results, metrics)

    log.info("Retrieval metrics computed")

    with open(config.MetricsFile, "w") as f:
        json.dump(metrics, f, indent=4)

    log.info(f"Metrics saved to {config.MetricsFile}")

    with open(config.RetrievalMetricsFile, "w") as f:
        json.dump(retrieval_metrics, f, indent=4)

    log.info(f"Retrieval metrics saved to {config.RetrievalMetricsFile}")
