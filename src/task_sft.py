import logging
from datasets import Dataset, interleave_datasets, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

logger = logging.getLogger(__name__)

def run_task_sft(cfg, dummy_data):
    model_name = cfg.get("model_name", "sshleifer/tiny-gpt2")
    max_seq_length = cfg.get("max_seq_length", 512)
    packing = cfg.get("packing", True)
    use_lora = cfg.get("use_lora", False)

    print(f"Initializing model {model_name} for Task SFT...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dummy_data:
        print("Using dummy dataset for Task SFT...")
        dummy_data_list = [
            {"messages": [{"role": "user", "content": "Task 1 question?"}, {"role": "assistant", "content": "Task 1 answer."}]},
            {"messages": [{"role": "user", "content": "Another specific task?"}, {"role": "assistant", "content": "Specific answer."}]}
        ] * 100
        ds = Dataset.from_list(dummy_data_list)
    else:
        print("Streaming SFT datasets based on config...")
        paths = cfg.get("dataset_paths", {})
        datasets = []
        probabilities = []

        if not paths:
             dataset_path = cfg.get("dataset_path")
             if dataset_path:
                 ds = load_dataset("json", data_files=f"{dataset_path}/*.jsonl", split="train")
             else:
                 raise ValueError("No datasets configured for Task SFT.")
        else:
            for lang, path in paths.items():
                try:
                    loaded_ds = load_dataset(path, split="train", streaming=True)
                    datasets.append(loaded_ds)
                    probabilities.append(1.0)
                except Exception as e:
                    print(f"Error loading {path}: {e}")

            if not datasets:
                raise ValueError("No datasets loaded successfully.")

            total = sum(probabilities)
            probabilities = [p/total for p in probabilities]
            ds = interleave_datasets(datasets, probabilities=probabilities, stopping_strategy="all_exhausted")

    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
        tokenizer.add_special_tokens({'additional_special_tokens': ['<|im_start|>', '<|im_end|>']})
        model.resize_token_embeddings(len(tokenizer))
        tokenizer.eos_token = '<|im_end|>'

    if use_lora:
        from peft import LoraConfig, get_peft_model
        print("Applying LoRA to the model...")
        lora_cfg = cfg.get("lora", {})
        peft_config = LoraConfig(
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            target_modules=lora_cfg.get("target_modules", ["c_attn"]),
            lora_dropout=lora_cfg.get("lora_dropout", 0.1),
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        )
        model = get_peft_model(model, peft_config)

    training_args = SFTConfig(
        output_dir=cfg.get("output_dir", "./task-sft-output"),
        learning_rate=cfg.get("learning_rate", 2e-5),
        num_train_epochs=cfg.get("epochs", 1),
        max_steps=cfg.get("max_steps", 1000),
        per_device_train_batch_size=cfg.get("batch_size", 4),
        save_steps=cfg.get("save_steps", 1000),
        logging_steps=cfg.get("logging_steps", 10),
        packing=packing,
        max_seq_length=max_seq_length,
        dataset_text_field="text"
    )


    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
    )

    print("Starting Task Supervised Fine-Tuning...")
    trainer.train()
    print("Task SFT completed successfully.")
