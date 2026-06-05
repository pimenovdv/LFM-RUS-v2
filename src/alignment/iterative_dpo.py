import json
import logging
import os
from typing import List, Dict, Any
from datasets import Dataset, load_dataset
from openai import OpenAI
from src.alignment.rlaif import generate_responses, evaluate_with_llm_judge

logger = logging.getLogger(__name__)

def run_iterative_dpo_iteration(cfg: Dict[str, Any], model, tokenizer, iteration: int, dummy_data: bool = False):
    print(f"Starting Iterative DPO generation for iteration {iteration}...")

    max_completion_length = cfg.get("max_completion_length", 128)
    output_dir = cfg.get("output_dir", "./iterative_dpo_output")
    rules = cfg.get("constitutional_rules", "Be helpful, harmless, and honest.")
    judge_model_name = cfg.get("judge_model_name", "gpt-4o-mini")
    api_key = cfg.get("openai_api_key", os.environ.get("OPENAI_API_KEY", ""))
    base_url = cfg.get("openai_base_url", None)

    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    if dummy_data:
        dataset = Dataset.from_dict({
            "prompt": ["What is 2+2?", "Write a function.", "Explain math"] * 10
        })
    else:
        dataset_path = cfg.get("dataset_path")
        if not dataset_path:
            raise ValueError("dataset_path must be provided in config for Iterative DPO")
        dataset = load_dataset(dataset_path, split="train")

    prompts = dataset["prompt"]

    print(f"Generating 2 responses per prompt for {len(prompts)} prompts...")
    all_completions = generate_responses(model, tokenizer, prompts, 2, max_completion_length)

    dpo_data = []
    print("Evaluating pairs with LLM Judge...")
    for prompt, completions in zip(prompts, all_completions):
        if len(completions) < 2:
             continue
        resp_a, resp_b = completions[0], completions[1]
        preference = evaluate_with_llm_judge(prompt, resp_a, resp_b, rules, client, judge_model_name)

        if preference == "A":
            chosen = resp_a
            rejected = resp_b
        else:
            chosen = resp_b
            rejected = resp_a

        dpo_data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    # Save to jsonl
    os.makedirs(output_dir, exist_ok=True)
    generated_data_path = os.path.join(output_dir, f"iterative_dpo_dataset_iter_{iteration}.jsonl")
    with open(generated_data_path, "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(dpo_data)} Iterative DPO instances to {generated_data_path}")
    return generated_data_path
