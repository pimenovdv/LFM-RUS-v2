import json
import logging
import os
from typing import List, Dict, Any
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = logging.getLogger(__name__)

def generate_rejected_responses(model, tokenizer, prompts: List[str], max_length: int) -> List[str]:
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto" if tokenizer.pad_token is not None else None)

    all_responses = []
    for prompt in prompts:
        outputs = generator(
            prompt,
            max_new_tokens=max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )
        responses = [out["generated_text"] for out in outputs]
        all_responses.append(responses[0])

    return all_responses

def run_spin_pipeline(cfg: Dict[str, Any], model, tokenizer, dummy_data: bool = False):
    print("Starting SPIN (Self-Play Fine-Tuning) data generation pipeline...")

    max_completion_length = cfg.get("max_completion_length", 128)
    output_dir = cfg.get("output_dir", "./spin_output")

    if dummy_data:
        dataset = Dataset.from_dict({
            "prompt": ["What is 2+2?", "Write a function.", "Explain math"],
            "chosen": ["It is 4.", "def func(): pass", "Math is logic."]
        })
    else:
        dataset_path = cfg.get("dataset_path")
        if not dataset_path:
            raise ValueError("dataset_path must be provided in config for SPIN")
        # Assume prompts are in 'prompt' column and real responses in 'chosen' or 'completion'
        dataset = load_dataset(dataset_path, split="train")

    prompts = dataset["prompt"]
    chosen = dataset["chosen"] if "chosen" in dataset.column_names else (dataset["completion"] if "completion" in dataset.column_names else dataset["text"])

    print(f"Generating rejected responses for {len(prompts)} prompts...")
    rejected_responses = generate_rejected_responses(model, tokenizer, prompts, max_completion_length)

    spin_data = []
    for p, c, r in zip(prompts, chosen, rejected_responses):
        spin_data.append({"prompt": p, "chosen": c, "rejected": r})

    # Save to jsonl
    os.makedirs(output_dir, exist_ok=True)
    generated_data_path = os.path.join(output_dir, "spin_dataset.jsonl")
    with open(generated_data_path, "w", encoding="utf-8") as f:
        for item in spin_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(spin_data)} SPIN instances to {generated_data_path}")
    return generated_data_path
