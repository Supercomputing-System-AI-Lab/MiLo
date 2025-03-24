import argparse
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation/lm_eval')))
from transformers import AutoTokenizer
from evaluation.lm_eval import evaluator
from evaluation.lm_eval.models.huggingface import HFLM
from evaluation.lm_eval.tasks import initialize_tasks
from MiLo.core.quantize import *
from transformers import AutoTokenizer
from MiLo.core.quantize import *
from MiLo.models.hf.mixtral import MixtralMiLo
from MiLo.models.hf.deepseek import DeepSeekMoEMiLo
from MiLo.core.quantize import *
LM_EVAL_TASK_KWARGS_DICT = {
    "mmlu": {"task": "mmlu", "num_fewshot": 5, "batch_size": 8, "metric": "acc"},
    "triQA": {"task": "triviaqa", "num_fewshot": 5, "batch_size": 16, "metric": "exact_match"},
}

AutoMiLoHFModel = MixtralMiLo

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--base_dir', type=str, required=True, help="base directory to save the quantized model")
    parser.add_argument('--model_id', type=str, required=True, help="base model type")
    args = parser.parse_args()

    print(f"Start few-shot evaluation on {args.base_dir}")
    
    if "Mixtral" in args.model_id:
        model_id = "mistralai/Mixtral-8x7B-v0.1" 
        AutoMiLoHFModel = MixtralMiLo
    elif "DeepSeek" in args.model_id:
        model_id = "deepseek-ai/deepseek-moe-16b-base"
        AutoMiLoHFModel = DeepSeekMoEMiLo
    else:
        NotImplementedError("This model is not implemented yet")

    quant_model_dir = f"{args.base_dir}/model"
    lorc_dir = f"{args.base_dir}/lorc"
    lorc_dtype = "int3"
    with open(f"{args.base_dir}/ranks.json", "r", encoding="utf-8") as f:
        ranks  = json.load(f)

    model = AutoMiLoHFModel.from_quantized(quant_model_dir,LoRC_weight_path=lorc_dir,
                                        LoRC_dtype = lorc_dtype,
                                        ranks=ranks)
    tokenizer    = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token =tokenizer.eos_token

    save_file_path = os.path.join(args.base_dir, "eval_result.json")

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

    print(">>>>> Results <<<<<")
    average = sum(v for v in all_metrics.values()) / len(all_metrics)
    all_metrics["average"] = average
    print(f"Metrics: {all_metrics}")

if __name__ == "__main__":
    main()
