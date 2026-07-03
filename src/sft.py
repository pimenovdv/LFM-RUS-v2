import logging
from datasets import Dataset, interleave_datasets, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration

import torch
from transformers import DataCollatorForLanguageModeling
import numpy as np


class SFTDiffusionDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mask_token_id, max_timesteps=1000, block_size=None, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.mask_token_id = mask_token_id
        self.max_timesteps = max_timesteps
        self.block_size = block_size

    def __call__(self, examples, return_tensors=None):
        batch = super().__call__(examples, return_tensors)
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        batch_size, seq_len = input_ids.shape

        # We need to find the <|im_start|>assistant markers to know where the answers are
        # If no such markers, we might just fall back to standard behavior or mask nothing
        assistant_start_token = self.tokenizer.convert_tokens_to_ids("<|im_start|>")

        timesteps = torch.zeros(batch_size, dtype=torch.long)

        for i in range(batch_size):
            # Find answer boundaries
            # A simple heuristic: look for assistant start
            # For a more robust approach, we need to decode or know the exact sequence.
            # Assuming standard ChatML: <|im_start|>assistant
            # Let's find all <|im_start|> tokens
            im_starts = (input_ids[i] == assistant_start_token).nonzero(as_tuple=True)[0]

            # We want to mask ONLY the tokens in the assistant's replies.
            answer_mask = torch.zeros(seq_len, dtype=torch.bool)

            # Since SFT can have multiple turns, we iterate through <|im_start|>
            for start_idx in im_starts:
                # check if next token is 'assistant' (we might just check if it's the assistant role)
                # But it's easier to just mask until <|im_end|> or EOS
                # Let's decode to check role, or assume all starts that have 'assistant' after it
                role_tokens = input_ids[i, start_idx+1:start_idx+3]
                decoded_role = self.tokenizer.decode(role_tokens).strip()
                if "assistant" in decoded_role.lower():
                    # Find end of turn
                    end_idx = seq_len
                    for j in range(start_idx + 1, seq_len):
                        if input_ids[i, j] == self.tokenizer.eos_token_id:
                            end_idx = j + 1
                            break
                    # Answer is from start_idx + length of `<|im_start|>assistant` to end_idx
                    # Just masking from start_idx+3 for safety (approximate length of role string)
                    answer_mask[start_idx+3:end_idx] = True

            labels[i, ~answer_mask] = -100

            if answer_mask.sum() > 0:
                # Sample t
                t = torch.rand(1).item()
                # Determine how many to mask
                num_to_mask = int(t * answer_mask.sum().item())
                # Get indices of answer tokens
                answer_indices = answer_mask.nonzero(as_tuple=True)[0]
                # Randomly select which to mask
                mask_indices = answer_indices[torch.randperm(len(answer_indices))[:num_to_mask]]

                # Replace with mask_token_id
                input_ids[i, mask_indices] = self.mask_token_id

                # Set labels to -100 for UNMASKED answer tokens (loss only on masked)
                unmasked_answer_indices = torch.tensor(list(set(answer_indices.tolist()) - set(mask_indices.tolist())), dtype=torch.long)
                if len(unmasked_answer_indices) > 0:
                    labels[i, unmasked_answer_indices] = -100

                # Compute timestep
                timesteps[i] = max(1, int(t * self.max_timesteps))
            else:
                timesteps[i] = 1 # fallback


        batch["input_ids"] = input_ids
        batch["labels"] = labels
        batch["timesteps"] = timesteps

        # Blockwise SFT Attention Mask
        if self.block_size is not None and self.block_size > 0:
            mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.float32)
            num_blocks = (seq_len + self.block_size - 1) // self.block_size
            for i in range(num_blocks):
                for j in range(num_blocks):
                    if j <= i: # causal between blocks
                        start_i = i * self.block_size
                        end_i = min((i + 1) * self.block_size, seq_len)
                        start_j = j * self.block_size
                        end_j = min((j + 1) * self.block_size, seq_len)
                        mask[:, :, start_i:end_i, start_j:end_j] = 1.0
            batch["attention_mask"] = mask

        return batch




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

    is_diffusion = isinstance(model, DiffusionModelForConditionalGeneration) or getattr(model.config, "model_type", "") == "lfm_masked_diffusion"

    if is_diffusion:
        packing = False
        print("Diffusion model detected. Disabled packing and using SFTDiffusionDataCollator.")

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

    data_collator = None
    if is_diffusion:
        block_diffusion_cfg = cfg.get("block_diffusion", {})
        block_size = block_diffusion_cfg.get("block_size", 64) if block_diffusion_cfg.get("enabled", False) else None
        data_collator = SFTDiffusionDataCollator(
            tokenizer=tokenizer,
            mask_token_id=getattr(model.config, "mask_token_id", tokenizer.pad_token_id),
            max_timesteps=getattr(model.config, "max_timesteps", 1000),
            block_size=block_size
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )



    print("Starting Supervised Fine-Tuning...")
    trainer.train()

    push_repo = cfg.get("push_to_hub")
    if push_repo:
        print(f"Pushing model to Hub: {push_repo}...")
        trainer.model.push_to_hub(push_repo, commit_message="Upload SFT model")
        trainer.processing_class.push_to_hub(push_repo, commit_message="Upload SFT model") if hasattr(trainer, "processing_class") and trainer.processing_class else trainer.tokenizer.push_to_hub(push_repo, commit_message="Upload SFT model") if hasattr(trainer, "tokenizer") and trainer.tokenizer else None

    print("SFT completed successfully.")
