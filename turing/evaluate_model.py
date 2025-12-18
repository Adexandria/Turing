import time

from datasets import DatasetDict
from loguru import logger
import numpy as np
import pandas as pd
import torch

import turing.config as config


def calculate_submission_score(avg_f1: float, avg_runtime: float, avg_flops: float) -> float:
    """
    Calculates the final competition score.
    The score is a weighted sum of F1 score, runtime, and GFLOPS.
    Weights:
    - F1 Score: 60%
    - Runtime: 20%
    - GFLOPS: 20%

    Args:
        avg_f1 (float): Average F1 score across all categories.
        avg_runtime (float): Average runtime in seconds.
        avg_flops (float): Average GFLOPS.

    Returns:
        float: Final submission score.
    """

    score_f1 = 0.6 * avg_f1

    runtime_ratio = (config.MAX_AVG_RUNTIME - avg_runtime) / config.MAX_AVG_RUNTIME
    score_runtime = 0.2 * max(runtime_ratio, 0)

    flops_ratio = (config.MAX_AVG_FLOPS - avg_flops) / config.MAX_AVG_FLOPS
    score_flops = 0.2 * max(flops_ratio, 0)

    total_score = score_f1 + score_runtime + score_flops

    logger.info(f"  F1 Score (60%): {score_f1:.4f} (avg_f1: {avg_f1:.4f})")
    logger.info(
        f"  Runtime Score (20%): {score_runtime:.4f} (avg_runtime: {avg_runtime:.4f}s / {config.MAX_AVG_RUNTIME}s)"
    )
    logger.info(
        f"  GFLOPS Score (20%): {score_flops:.4f} (avg_flops: {avg_flops:.4f} / {config.MAX_AVG_FLOPS})"
    )
    logger.info("  ====================")
    logger.info(f"  Final Score: {total_score:.4f}")

    return total_score


def evaluate_models(models: dict, dataset: DatasetDict):
    """
    Evaluates the provided models on the test datasets for each language.
    Computes precision, recall, and F1 score for each category and language.
    Also measures average runtime and GFLOPS for model inference.

    Args:
        models (dict): A dictionary mapping language codes to their respective models.
        dataset (DatasetDict): A DatasetDict containing test datasets for each language.

    Returns:
        pd.DataFrame: DataFrame containing precision, recall, and F1 scores for each category and language.
        float: Final submission score calculated based on average F1, runtime, and GF
    """

    total_flops = 0
    total_time = 0
    scores = []

    for lan in config.LANGS:
        logger.info(f"\n--- Evaluating Language: {lan.upper()} ---")
        model = models[lan]

        with torch.profiler.profile(with_flops=True) as p:
            test_data = dataset[f"{lan}_test"]
            x = test_data[config.INPUT_COLUMN]
            x = list(x) if hasattr(x, 'tolist') else x  # Convert pandas Series to list
            y_true = np.array(test_data[config.LABEL_COLUMN]).T

            begin = time.time()
            for i in range(10):
                y_pred = model.predict(x)
                y_pred = np.asarray(y_pred).T
            total = time.time() - begin
            total_time = total_time + total

        total_flops = total_flops + (sum(k.flops for k in p.key_averages()) / 1e9)

        for i in range(len(y_pred)):
            assert len(y_pred[i]) == len(y_true[i])
            tp = sum([true == pred == 1 for (true, pred) in zip(y_true[i], y_pred[i])])
            #tn = sum([true == pred == 0 for (true, pred) in zip(y_true[i], y_pred[i])])
            fp = sum([true == 0 and pred == 1 for (true, pred) in zip(y_true[i], y_pred[i])])
            fn = sum([true == 1 and pred == 0 for (true, pred) in zip(y_true[i], y_pred[i])])
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = (2 * tp) / (2 * tp + fp + fn)
            scores.append({
                "lan": lan,
                "cat": config.LABELS_MAP[lan][i],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })

    logger.info(f"Compute in GFLOPs: {total_flops / 10}")
    logger.info(f"Avg runtime in seconds: {total_time / 10}")
    scores = pd.DataFrame(scores)
    print(scores)

    avg_f1 = scores["f1"].mean()
    avg_runtime = total_time / 10
    avg_flops = total_flops / 10

    final_score = calculate_submission_score(avg_f1, avg_runtime, avg_flops)

    logger.info(f"Final Score for {lan.upper()}: {final_score:.4f}")

    return scores, final_score
