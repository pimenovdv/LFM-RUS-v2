import logging
from datasets import Dataset, interleave_datasets, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

logger = logging.getLogger(__name__)

def run_sft(cfg, dummy_data):
    model_name = cfg.get("model_name", "sshleifer/tiny-gpt2")
    max_seq_length = cfg.get("max_seq_length", 512)
    packing = cfg.get("packing", True)

    print(f"Initializing model {model_name} for SFT...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Ensuring chat template exists for ChatML if not using basic completion
    # TinyGPT-2 doesn't have a chat template by default, but SFTTrainer handles basic formatting
    # or expects "messages" column for standard chat formats if a template is set.
    # For general instruction SFT, we often just use text completions or messages.

    if dummy_data:
        print("Using dummy dataset for SFT...")
        # Simulate ChatML style or basic Q/A format
        dummy_data_list = [
            {"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4."}]},
            {"messages": [{"role": "user", "content": "Write hello world in Python."}, {"role": "assistant", "content": "print('hello world')"}]},
            {"messages": [{"role": "user", "content": "Привет, как дела?"}, {"role": "assistant", "content": "Привет! Все отлично."}]}
        ] * 100
        ds = Dataset.from_list(dummy_data_list)
    else:
        print("Streaming SFT datasets based on config...")
        paths = cfg.get("dataset_paths", {})
        datasets = []
        probabilities = []

        # Typically SFT config might have different datasets or ratios, here using paths
        if not paths:
             print("Warning: No dataset_paths specified in config, attempting to load default 'json' or use generic logic.")
             # Fallback if config is minimal
             dataset_path = cfg.get("dataset_path")
             if dataset_path:
                 ds = load_dataset("json", data_files=f"{dataset_path}/*.jsonl", split="train")
             else:
                 raise ValueError("No datasets configured for SFT.")
        else:
            for lang, path in paths.items():
                try:
                    loaded_ds = load_dataset(path, split="train", streaming=True)
                    datasets.append(loaded_ds)
                    # Simple equal probability if not specified
                    probabilities.append(1.0)
                except Exception as e:
                    print(f"Error loading {path}: {e}")

            if not datasets:
                raise ValueError("No datasets loaded successfully.")

            total = sum(probabilities)
            probabilities = [p/total for p in probabilities]
            ds = interleave_datasets(datasets, probabilities=probabilities, stopping_strategy="all_exhausted")

    model = AutoModelForCausalLM.from_pretrained(model_name)

    training_args = SFTConfig(
        output_dir=cfg.get("output_dir", "./sft-output"),
        learning_rate=cfg.get("learning_rate", 2e-5),
        num_train_epochs=cfg.get("epochs", 1),
        max_steps=cfg.get("max_steps", 1000),
        per_device_train_batch_size=cfg.get("batch_size", 4),
        save_steps=cfg.get("save_steps", 1000),
        logging_steps=cfg.get("logging_steps", 10),
        packing=packing,
        max_seq_length=max_seq_length,
        dataset_text_field="text" # default, will be overridden if we pass dataset format
    )

    # For SFT with ChatML (messages format), trl's SFTTrainer natively supports it if dataset has 'messages' column
    # If the tokenizer doesn't have a chat template, SFTTrainer might complain, so let's set a simple one if missing.
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
        # We must add custom tokens for EOS control as requested by general_sft.md
        tokenizer.add_special_tokens({'additional_special_tokens': ['<|im_start|>', '<|im_end|>']})
        model.resize_token_embeddings(len(tokenizer))
        # Ensure EOS control: tie padding/eos behavior to explicit generation boundaries
        tokenizer.eos_token = '<|im_end|>'

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
    )


    print("Starting Supervised Fine-Tuning...")
    trainer.train()

    push_repo = cfg.get("push_to_hub")
    if push_repo:
        print(f"Pushing model to Hub: {push_repo}...")
        trainer.model.push_to_hub(push_repo, commit_message="Upload SFT model")
        trainer.processing_class.push_to_hub(push_repo, commit_message="Upload SFT model") if hasattr(trainer, "processing_class") and trainer.processing_class else trainer.tokenizer.push_to_hub(push_repo, commit_message="Upload SFT model") if hasattr(trainer, "tokenizer") and trainer.tokenizer else None

    print("SFT completed successfully.")
