from typing import Dict, Any, List
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.orpo import ORPOTrainer, ORPOConfig
from trl import DPOTrainer, DPOConfig, GRPOTrainer, GRPOConfig, KTOTrainer, KTOConfig

def accuracy_reward(prompts: List[str], completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
    """
    Reward function that checks if the completion contains a <solution> tag
    and returns a positive reward if found, otherwise negative.
    (Placeholder for a real accuracy reward)
    """
    rewards = []
    # GRPO format has lists of dicts
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict) else str(completion)
        if "<solution>" in text:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards

def variance_reward(prompts: List[str], completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
    """
    Reward function that penalizes loops or low variance responses.
    Checks for repeating n-grams.
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict) else str(completion)
        # Simple loop detection: repeating sequences
        words = text.split()
        if len(words) < 10:
             rewards.append(0.0)
             continue

        # Check for 3-gram repetitions
        ngrams = set()
        has_loop = False
        for i in range(len(words) - 2):
            ngram = tuple(words[i:i+3])
            if ngram in ngrams:
                has_loop = True
                break
            ngrams.add(ngram)

        if has_loop:
            rewards.append(-2.0)
        else:
            rewards.append(0.5)

    return rewards

def format_dpo_dataset(dataset_path: str, split: str = "train") -> Dataset:
    """Loads and formats a DPO dataset expecting 'prompt', 'chosen', and 'rejected' columns."""
    dataset = load_dataset(dataset_path, split=split)
    return dataset


def format_kto_dataset(dataset_path: str, split: str = "train") -> Dataset:
    """Loads and formats a KTO dataset expecting 'prompt', 'completion', and 'label' columns."""
    dataset = load_dataset(dataset_path, split=split)
    return dataset

def format_grpo_dataset(dataset_path: str, split: str = "train") -> Dataset:
    """Loads and formats a GRPO dataset expecting 'prompt' column."""
    dataset = load_dataset(dataset_path, split=split)
    return dataset

def run_alignment_pipeline(cfg: Dict[str, Any], dummy_data: bool = False):
    """
    Runs the alignment pipeline (DPO or GRPO) based on the configuration.
    """
    method = cfg.get("method", "dpo").lower()
    model_name = cfg.get("model_name", "sshleifer/tiny-gpt2")

    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    output_dir = cfg.get("output_dir", f"./alignment-{method}-output")
    batch_size = cfg.get("batch_size", 4)
    learning_rate = float(cfg.get("learning_rate", 1e-5))
    epochs = cfg.get("epochs", 1)

    if method in ["dpo", "ipo"]:
        print(f"Starting {method.upper()} training...")
        if dummy_data:
            dataset = Dataset.from_dict({
                "prompt": ["How to write a function?", "Explain math"],
                "chosen": ["def func(): pass", "Math is logic"],
                "rejected": ["func func func func", "math math math"]
            })
        else:
            dataset_path = cfg.get("dataset_path")
            if not dataset_path:
                raise ValueError("dataset_path must be provided in config for DPO")
            dataset = format_dpo_dataset(dataset_path)

        dpo_config = DPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            loss_type=cfg.get("loss_type", "ipo" if method == "ipo" else "sigmoid"),
            max_length=cfg.get("max_prompt_length", 512) + cfg.get("max_completion_length", 1024),
            remove_unused_columns=cfg.get("remove_unused_columns", False),
            use_cpu=cfg.get("use_cpu", True), # to avoid bf16 errors on test instances
            bf16=cfg.get("bf16", False),
            fp16=cfg.get("fp16", False)
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=None, # TRL will create a reference model automatically if None
            args=dpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        trainer.train()
        print(f"{method.upper()} training completed.")

    elif method == "grpo":
        print("Starting GRPO (RLVR) training...")
        if dummy_data:
            dataset = Dataset.from_dict({
                "prompt": ["Write a python function that adds two numbers", "Calculate 2 + 2"]
            })
        else:
            dataset_path = cfg.get("dataset_path")
            if not dataset_path:
                raise ValueError("dataset_path must be provided in config for GRPO")
            dataset = format_grpo_dataset(dataset_path)

        reward_funcs_names = cfg.get("reward_funcs", ["accuracy", "variance"])
        reward_funcs = []
        for name in reward_funcs_names:
            if name == "accuracy":
                reward_funcs.append(accuracy_reward)
            elif name == "variance":
                reward_funcs.append(variance_reward)
            else:
                print(f"Warning: Unknown reward function {name}")

        grpo_config = GRPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            max_completion_length=cfg.get("max_completion_length", 1024),
            remove_unused_columns=cfg.get("remove_unused_columns", False),
            use_cpu=cfg.get("use_cpu", True),
            bf16=cfg.get("bf16", False),
            fp16=cfg.get("fp16", False),
            num_generations=cfg.get("num_generations", batch_size),
        )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        trainer.train()
        print("GRPO training completed.")

    elif method == "kto":
        print("Starting KTO training...")
        if dummy_data:
            dataset = Dataset.from_dict({
                "prompt": ["Write a function", "Explain math"],
                "completion": ["def func(): pass", "math math math"],
                "label": [True, False]
            })
        else:
            dataset_path = cfg.get("dataset_path")
            if not dataset_path:
                raise ValueError("dataset_path must be provided in config for KTO")
            dataset = format_kto_dataset(dataset_path)

        kto_config = KTOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            max_length=cfg.get("max_prompt_length", 512) + cfg.get("max_completion_length", 1024),

            remove_unused_columns=cfg.get("remove_unused_columns", False),
            use_cpu=cfg.get("use_cpu", True),
            bf16=cfg.get("bf16", False),
            fp16=cfg.get("fp16", False)
        )

        trainer = KTOTrainer(
            model=model,
            args=kto_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        trainer.train()
        print("KTO training completed.")


    elif method == "orpo":
        print("Starting ORPO training...")
        if dummy_data:
            dataset = Dataset.from_dict({
                "prompt": ["How to write a function?", "Explain math"],
                "chosen": ["def func(): pass", "Math is logic"],
                "rejected": ["func func func func", "math math math"]
            })
        else:
            dataset_path = cfg.get("dataset_path")
            if not dataset_path:
                raise ValueError("dataset_path must be provided in config for ORPO")
            dataset = format_dpo_dataset(dataset_path)

        orpo_config = ORPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            max_length=cfg.get("max_prompt_length", 512) + cfg.get("max_completion_length", 1024),
            remove_unused_columns=cfg.get("remove_unused_columns", False),
            use_cpu=cfg.get("use_cpu", True),
            bf16=cfg.get("bf16", False),
            fp16=cfg.get("fp16", False)
        )

        trainer = ORPOTrainer(
            model=model,
            args=orpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        trainer.train()
        print("ORPO training completed.")
    else:
        raise ValueError(f"Unknown alignment method: {method}. Use 'dpo', 'ipo', 'kto', 'grpo' or 'orpo'.")


    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

    push_repo = cfg.get("push_to_hub")
    if push_repo:
        print(f"Pushing model to Hub: {push_repo}...")
        trainer.model.push_to_hub(push_repo, commit_message=f"Upload Alignment ({method}) model")
        trainer.processing_class.push_to_hub(push_repo, commit_message=f"Upload Alignment ({method}) model") if hasattr(trainer, "processing_class") and trainer.processing_class else trainer.tokenizer.push_to_hub(push_repo, commit_message=f"Upload Alignment ({method}) model") if hasattr(trainer, "tokenizer") and trainer.tokenizer else None