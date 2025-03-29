import time
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation/lm_eval')))
from evaluation.lm_eval import evaluator
from evaluation.lm_eval.models.huggingface import HFLM
from evaluation.lm_eval.tasks import initialize_tasks


LM_EVAL_TASK_KWARGS_DICT = {
    "mmlu": {"task": "mmlu", "num_fewshot": 5, "batch_size": 8, "metric": "acc"},
    "triQA": {"task": "triviaqa", "num_fewshot": 5, "batch_size": 16, "metric": "exact_match"},
}


def eval_fewshots(model,tokenizer,result_save_path):
    begin = time.time()
    print(f"Start few-shot evaluation")
    save_file_path = os.path.join(result_save_path, "eval_result.json")
    all_metrics = {}
    if os.path.exists(save_file_path):
        with open(save_file_path, 'r') as file:
            all_metrics = json.load(file)

    for task_kwargs in LM_EVAL_TASK_KWARGS_DICT.values():
        print(f"Evaluating task: {task_kwargs['task']}")
        task_name = task_kwargs["task"]
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=task_kwargs["batch_size"],
        )
        initialize_tasks(verbosity="ERROR")
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=task_name,
            num_fewshot=task_kwargs["num_fewshot"],
            batch_size=task_kwargs["batch_size"],
            log_samples=False,
        )
        metric = task_kwargs["metric"]
        for key, value in results["results"][task_name].items():
            if key.startswith(metric + ","):
                all_metrics[f"{task_name}_{metric}"] = value

        with open(save_file_path, 'w') as file:
            json.dump(all_metrics, file, indent=4)

    end = time.time()
    print(">>>>> Results <<<<<")
    # average = sum(v for v in all_metrics.values()) / len(all_metrics)
    # all_metrics["average"] = average
    print(f"Metrics: {all_metrics}")
    print(f"Evaluation time: {end - begin:.2f}s")

