#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的评估结果保存工具。
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


@dataclass
class EvalMeta:
    """评估元信息数据类"""

    model_name: str
    dataset: str
    seed: int
    extra_config: Optional[Dict[str, Any]] = None


def save_evaluation_results(
    result_dir: str,
    meta: EvalMeta,
    report: Dict[str, Any],
    y_true: Iterable[int],
    y_pred: Iterable[int],
) -> None:
    """
    将实验评估结果保存为标准化的 json 文件。
    """
    os.makedirs(result_dir, exist_ok=True)

    y_true_arr = np.array(list(y_true))
    y_pred_arr = np.array(list(y_pred))

    cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()

    eval_results = {
        "meta": asdict(meta),
        "timestamp": datetime.now().isoformat(),
        "test_accuracy": report.get("accuracy"),
        "test_f1_macro": report.get("macro avg", {}).get("f1-score"),
        "test_f1_weighted": report.get("weighted avg", {}).get("f1-score"),
        "classification_report": report,
        "confusion_matrix": cm,
        "predictions": y_pred_arr.tolist(),
        "true_labels": y_true_arr.tolist(),
    }

    out_path = os.path.join(result_dir, "evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
