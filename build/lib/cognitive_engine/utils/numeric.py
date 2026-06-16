from __future__ import annotations

import random
from typing import Dict, List


OPERATIONS = {"add": 0, "sub": 1, "mul": 2}


def apply_operation(a: float, b: float, operation: str) -> float:
    if operation == "add":
        return a + b
    if operation == "sub":
        return a - b
    if operation == "mul":
        return a * b
    raise ValueError(f"Unsupported operation: {operation}")


def build_numeric_features(a: float, b: float, scale: float) -> List[float]:
    left = a / scale
    right = b / scale
    return [
        left,
        right,
        left * left,
        right * right,
        left * right,
        left - right,
        abs(left - right),
    ]


def make_numeric_sample(a: int, b: int, operation: str, scale: float) -> Dict[str, float]:
    return {
        "a": float(a),
        "b": float(b),
        "operation": operation,
        "operation_id": OPERATIONS[operation],
        "target": float(apply_operation(a, b, operation)),
        "features": build_numeric_features(a, b, scale),
    }


def generate_numeric_dataset(max_operand: int, operations: List[str], size: int, seed: int, scale: float | None = None) -> List[Dict[str, float]]:
    rng = random.Random(seed)
    scale = scale if scale is not None else max_operand
    dataset = []
    for _ in range(size):
        a = rng.randint(0, max_operand)
        b = rng.randint(0, max_operand)
        operation = rng.choice(operations)
        dataset.append(make_numeric_sample(a, b, operation, scale))
    return dataset


def enumerate_numeric_dataset(max_operand: int, operations: List[str], seed: int, scale: float | None = None) -> List[Dict[str, float]]:
    scale = scale if scale is not None else max_operand
    dataset = [
        make_numeric_sample(a, b, operation, scale)
        for operation in operations
        for a in range(max_operand + 1)
        for b in range(max_operand + 1)
    ]
    rng = random.Random(seed)
    rng.shuffle(dataset)
    return dataset
