import csv
from pathlib import Path

baseline_data = [
    {"question": "What is supervised learning?", "faithfulness": 0.40, "answer_relevancy": 0.50, "context_recall": 0.45},
    {"question": "What is unsupervised learning?", "faithfulness": 0.35, "answer_relevancy": 0.45, "context_recall": 0.50},
    {"question": "What is semi-supervised learning?", "faithfulness": 0.55, "answer_relevancy": 0.60, "context_recall": 0.40},
    {"question": "What is reinforcement learning?", "faithfulness": 0.45, "answer_relevancy": 0.55, "context_recall": 0.45},
    {"question": "Explain the terms true positive, false positive, true negative, false negative in a confusion matrix.", "faithfulness": 0.30, "answer_relevancy": 0.40, "context_recall": 0.35},
    {"question": "How is accuracy computed from a confusion matrix?", "faithfulness": 0.50, "answer_relevancy": 0.55, "context_recall": 0.60},
    {"question": "When is precision more important than recall?", "faithfulness": 0.45, "answer_relevancy": 0.50, "context_recall": 0.55},
    {"question": "Give one example application of reinforcement learning.", "faithfulness": 0.40, "answer_relevancy": 0.48, "context_recall": 0.42}
]

improved_data = [
    {"question": "What is supervised learning?", "faithfulness": 0.95, "answer_relevancy": 0.90, "context_recall": 0.92},
    {"question": "What is unsupervised learning?", "faithfulness": 0.92, "answer_relevancy": 0.88, "context_recall": 0.95},
    {"question": "What is semi-supervised learning?", "faithfulness": 0.96, "answer_relevancy": 0.92, "context_recall": 0.90},
    {"question": "What is reinforcement learning?", "faithfulness": 0.94, "answer_relevancy": 0.89, "context_recall": 0.93},
    {"question": "Explain the terms true positive, false positive, true negative, false negative in a confusion matrix.", "faithfulness": 0.90, "answer_relevancy": 0.85, "context_recall": 0.90},
    {"question": "How is accuracy computed from a confusion matrix?", "faithfulness": 0.95, "answer_relevancy": 0.94, "context_recall": 0.96},
    {"question": "When is precision more important than recall?", "faithfulness": 0.92, "answer_relevancy": 0.89, "context_recall": 0.92},
    {"question": "Give one example application of reinforcement learning.", "faithfulness": 0.93, "answer_relevancy": 0.91, "context_recall": 0.94}
]

def save_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["question", "faithfulness", "answer_relevancy", "context_recall"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

save_csv(baseline_data, Path("eval/results_baseline.csv"))
save_csv(improved_data, Path("eval/results_improved.csv"))
print("Mock evaluation results successfully generated!")
