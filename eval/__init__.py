"""评测体系。

四个维度里只有 factual_accuracy 用 LLM-as-Judge，
其余三个（refusal_calibration / retrieval_efficiency / evidence_coverage）
都是确定性计算——评测本身的方差必须可控，否则分数波动会淹没真实的质量变化。

注意：本包名与内置函数 eval 同名，但作为包导入（from eval.evaluate import ...）
不会遮蔽内置函数，因为 Python 的名称查找对模块和内置函数走的是不同路径。
"""

from eval.evaluate import DIMENSION_WEIGHTS, PASS_THRESHOLD, Evaluator, load_test_cases

__all__ = ["DIMENSION_WEIGHTS", "PASS_THRESHOLD", "Evaluator", "load_test_cases"]
