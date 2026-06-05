from src.alignment.rewards.rewards import get_reward_function
import json
import logging
import os
from typing import List, Dict, Any
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.sft import run_sft

logger = logging.getLogger(__name__)

def generate_responses(model, tokenizer, prompts: List[str], n: int, max_length: int) -> List[List[str]]:
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto" if tokenizer.pad_token is not None else None)

    all_responses = []
    for prompt in prompts:
        # Generate n responses per prompt
        # do_sample=True, temperature>0 for diversity
        outputs = generator(
            prompt,
            max_new_tokens=max_length,
            num_return_sequences=n,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )
        responses = [out["generated_text"] for out in outputs]
        all_responses.append(responses)

    return all_responses


def run_rejection_sampling(cfg: Dict[str, Any], dummy_data: bool = False):
    print("Starting Rejection Sampling (Best-of-N) pipeline...")

    model_name = cfg.get("model_name", "sshleifer/tiny-gpt2")
    n_generations = cfg.get("num_generations", 4)
    max_completion_length = cfg.get("max_completion_length", 128)
    output_dir = cfg.get("output_dir", "./rejection_sampling_output")
    reward_func = cfg.get("reward_funcs", ["accuracy"])[0]

    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if dummy_data:
        prompts = ["What is 2+2?", "Write a function.", "Explain math"]
    else:
        dataset_path = cfg.get("dataset_path")
        if not dataset_path:
            raise ValueError("dataset_path must be provided in config for Rejection Sampling")
        # Assume prompts are in 'prompt' column
        ds = load_dataset(dataset_path, split="train")
        prompts = ds["prompt"]

    print(f"Generating {n_generations} responses for {len(prompts)} prompts...")
    all_completions = generate_responses(model, tokenizer, prompts, n_generations, max_completion_length)

    best_completions = []
    print("Evaluating responses...")
    reward_fn = get_reward_function(reward_func, cfg.get("reward_config", {}))
    for prompt, completions in zip(prompts, all_completions):
        try:
            scores = reward_fn(completions, [prompt] * len(completions))
        except TypeError:
            # Fallback for older reward functions that don't accept prompts
            scores = reward_fn(completions)
        best_idx = scores.index(max(scores))
        best_completions.append({"prompt": prompt, "completion": completions[best_idx]})

    # Save to jsonl
    os.makedirs(output_dir, exist_ok=True)
    generated_data_path = os.path.join(output_dir, "best_of_n.jsonl")
    with open(generated_data_path, "w", encoding="utf-8") as f:
        for item in best_completions:
            # Reformat to match SFT expectations or general format
            f.write(json.dumps({"text": f"{item['prompt']} {item['completion']}"}) + "\n")

    print(f"Saved {len(best_completions)} best responses to {generated_data_path}")

    # Run SFT on the generated dataset
    sft_cfg = cfg.copy()
    sft_cfg["dataset_path"] = output_dir  # SFT pipeline loads jsonl from this directory
    sft_cfg["dataset_paths"] = {} # Override to force loading from dataset_path

    print("Proceeding to SFT training with the best responses...")
    run_sft(sft_cfg, dummy_data=False) # Always false to load the generated dataset

    print("Rejection Sampling pipeline completed.")
