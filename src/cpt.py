import logging
from datasets import Dataset, interleave_datasets, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

logger = logging.getLogger(__name__)


import os
import glob
import torch
import safetensors.torch

def merge_top_k_checkpoints(output_dir: str, k: int):
    logger.info(f"Merging top {k} checkpoints from {output_dir}")
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        logger.warning("No checkpoints found to merge.")
        return

    # Sort checkpoints by step number
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    top_k_checkpoints = checkpoints[-k:]

    if len(top_k_checkpoints) < k:
        logger.warning(f"Requested to merge {k} checkpoints but only found {len(top_k_checkpoints)}.")

    merged_state_dict = {}
    num_ckpts = len(top_k_checkpoints)

    for ckpt_dir in top_k_checkpoints:
        logger.info(f"Loading checkpoint: {ckpt_dir}")
        model_file = os.path.join(ckpt_dir, "model.safetensors")
        if os.path.exists(model_file):
            state_dict = safetensors.torch.load_file(model_file)
        else:
            model_file = os.path.join(ckpt_dir, "pytorch_model.bin")
            if os.path.exists(model_file):
                state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
            else:
                logger.error(f"No valid model weights found in {ckpt_dir}")
                continue

        for key, tensor in state_dict.items():
            if key not in merged_state_dict:
                merged_state_dict[key] = tensor.clone().to(torch.float32) / num_ckpts
            else:
                merged_state_dict[key] += tensor.to(torch.float32) / num_ckpts

    merged_dir = os.path.join(output_dir, "merged_model")
    os.makedirs(merged_dir, exist_ok=True)
    safetensors.torch.save_file(merged_state_dict, os.path.join(merged_dir, "model.safetensors"))
    logger.info(f"Merged model saved to {merged_dir}")


class BlockDiffusionDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, block_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size

    def __call__(self, examples, return_tensors=None):
        batch = super().__call__(examples, return_tensors)
        seq_length = batch["input_ids"].shape[1]
        batch_size = batch["input_ids"].shape[0]

        # Ensure seq_length is a multiple of block_size, or just use block_size
        mask = torch.zeros((batch_size, 1, seq_length, seq_length), dtype=torch.float32)

        num_blocks = (seq_length + self.block_size - 1) // self.block_size
        for i in range(num_blocks):
            for j in range(num_blocks):
                if j <= i:
                    start_i = i * self.block_size
                    end_i = min((i + 1) * self.block_size, seq_length)
                    start_j = j * self.block_size
                    end_j = min((j + 1) * self.block_size, seq_length)
                    mask[:, :, start_i:end_i, start_j:end_j] = 1.0

        batch["attention_mask"] = mask
        return batch

def run_cpt(cfg, dummy_data):

    model_name = cfg.get("model_name", "sshleifer/tiny-gpt2")
    max_seq_length = cfg.get("max_seq_length", 512)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dummy_data:
        print("Using dummy dataset for CPT...")
        dummy_texts = [
            "This is a dummy text for continual pre-training.",
            "Let's write some code: print('Hello World')",
            "Math is fun: 2 + 2 = 4",
            "Тестовое предложение на русском языке."
        ] * 100

        ds = Dataset.from_dict({"text": dummy_texts})
    else:
        print("Streaming datasets based on config ratios and interleaving...")
        ratios = cfg.get("dataset_ratios", {"ru": 0.40, "en": 0.30, "code": 0.15, "math": 0.15})
        paths = cfg.get("dataset_paths", {})

        datasets = []
        probabilities = []

        for lang, ratio in ratios.items():
            path = paths.get(lang)
            if not path:
                print(f"Warning: No dataset path provided for '{lang}'. Skipping.")
                continue

            try:
                # Assuming the datasets are in JSONL format or standard HF hub format
                ds = load_dataset(path, split="train", streaming=True)
                datasets.append(ds)
                probabilities.append(ratio)
            except Exception as e:
                print(f"Error loading {path}: {e}")

        if not datasets:
            raise ValueError("No datasets loaded successfully.")

        total = sum(probabilities)
        probabilities = [p/total for p in probabilities]

        ds = interleave_datasets(datasets, probabilities=probabilities, stopping_strategy="all_exhausted")

    # Tokenize and group texts
    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"])
        res = {"input_ids": tokenized["input_ids"]}
        if "attention_mask" in tokenized:
            res["attention_mask"] = tokenized["attention_mask"]
        return res

    # The mapping should handle arbitrary batch sizes properly and drop all original columns
    if dummy_data:
        tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=ds.column_names)
    else:
        tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=list(ds.features.keys()) if hasattr(ds, "features") and ds.features else None)

    def group_texts(examples):
        # Concatenate all texts.
        keys_to_group = ["input_ids", "attention_mask"]
        concatenated_examples = {k: sum(examples[k], []) for k in keys_to_group if k in examples}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # Drop the small remainder
        total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_seq_length.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    # Initialize Model
    print(f"Initializing model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # WSD configuration
    wsd_cfg = cfg.get("wsd", {})
    if wsd_cfg.get("enabled", False):
        training_args_kwargs = {
            "lr_scheduler_type": "warmup_stable_decay",
            "lr_scheduler_kwargs": {"num_decay_steps": wsd_cfg.get("decay_steps", 0)},
            "warmup_ratio": wsd_cfg.get("warmup_ratio", 0.0)
        }
        if "warmup_steps" in wsd_cfg:
            training_args_kwargs["warmup_steps"] = wsd_cfg["warmup_steps"]
            training_args_kwargs.pop("warmup_ratio", None)
    else:
        training_args_kwargs = {}

    # Setup Training Arguments
    training_args = TrainingArguments(
        output_dir=cfg.get("output_dir", "./cpt-output"),
        learning_rate=cfg.get("learning_rate", 1e-4),
        num_train_epochs=cfg.get("epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        save_steps=cfg.get("save_steps", 1000),
        logging_steps=cfg.get("logging_steps", 10),
        save_total_limit=cfg.get("save_total_limit", 2),
        do_train=True,
        **training_args_kwargs
    )

    block_diffusion_cfg = cfg.get("block_diffusion", {})
    if block_diffusion_cfg.get("enabled", False):
        print(f"Using BlockDiffusionDataCollator with block_size={block_diffusion_cfg.get('block_size', 64)}")
        data_collator = BlockDiffusionDataCollator(
            block_size=block_diffusion_cfg.get("block_size", 64),
            tokenizer=tokenizer,
            mlm=False
        )
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    embedding_warmup_cfg = cfg.get("embedding_warmup", {})
    if embedding_warmup_cfg.get("enabled", False):
        print("Starting Embedding Warm-up Phase...")

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze input embeddings
        input_embeddings = model.get_input_embeddings()
        if input_embeddings is not None:
            for param in input_embeddings.parameters():
                param.requires_grad = True

        # Unfreeze output embeddings
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None:
            for param in output_embeddings.parameters():
                param.requires_grad = True

        warmup_args = TrainingArguments(
            output_dir=cfg.get("output_dir", "./cpt-output") + "-warmup",
            learning_rate=embedding_warmup_cfg.get("learning_rate", 1e-3),
            num_train_epochs=embedding_warmup_cfg.get("epochs", 1),
            per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
            save_steps=cfg.get("save_steps", 1000),
            logging_steps=cfg.get("logging_steps", 10),
            save_total_limit=1,
            do_train=True
        )

        warmup_trainer = Trainer(
            model=model,
            args=warmup_args,
            train_dataset=lm_datasets,
            data_collator=data_collator,
        )

        warmup_trainer.train()
        print("Embedding Warm-up Phase completed.")

        # Unfreeze all parameters for main CPT phase
        for param in model.parameters():
            param.requires_grad = True

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets,
        data_collator=data_collator,
    )


    print("Starting Continual Pre-Training...")
    trainer.train()

    push_repo = cfg.get("push_to_hub")
    if push_repo:
        print(f"Pushing model to Hub: {push_repo}...")
        trainer.model.push_to_hub(push_repo, commit_message="Upload CPT model")
        trainer.processing_class.push_to_hub(push_repo, commit_message="Upload CPT model") if hasattr(trainer, "processing_class") and trainer.processing_class else trainer.tokenizer.push_to_hub(push_repo, commit_message="Upload CPT model") if hasattr(trainer, "tokenizer") and trainer.tokenizer else None

    print("CPT completed successfully.")

    merge_top_k = cfg.get("merge_top_k", 0)
    if merge_top_k > 1:
        merge_top_k_checkpoints(cfg.get("output_dir", "./cpt-output"), merge_top_k)
