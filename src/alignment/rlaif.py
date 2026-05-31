import json
import logging
import os
from typing import List, Dict, Any
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from openai import OpenAI

logger = logging.getLogger(__name__)

def generate_responses(model, tokenizer, prompts: List[str], n: int, max_length: int) -> List[List[str]]:
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto" if tokenizer.pad_token is not None else None)

    all_responses = []
    for prompt in prompts:
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

def evaluate_with_llm_judge(prompt: str, response_a: str, response_b: str, rules: str, client: OpenAI, model_name: str) -> str:
    system_prompt = (
        "You are an impartial AI judge evaluating two responses to a user prompt. "
        "You must choose the best response based on the following constitutional rules:\n\n"
        f"{rules}\n\n"
        "Analyze both responses and decide which one better follows the rules and is more helpful. "
        "You must output ONLY valid JSON containing a single key 'preference' with value either 'A' or 'B'."
    )
    user_prompt = f"Prompt: {prompt}\n\nResponse A: {response_a}\n\nResponse B: {response_b}\n\nWhich response is better?"

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "preference_evaluation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "preference": {
                        "type": "string",
                        "enum": ["A", "B"]
                    }
                },
                "required": ["preference"],
                "additionalProperties": False
            }
        }
    }

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=schema,
            temperature=0.0
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        return parsed.get("preference", "A")
    except Exception as e:
        logger.error(f"LLM Judge error: {e}")
        return "A"  # Default to A on error to keep pipeline going

def run_rlaif_pipeline(cfg: Dict[str, Any], model, tokenizer, dummy_data: bool = False):
    print("Starting RLAIF (Constitutional AI) data generation pipeline...")

    max_completion_length = cfg.get("max_completion_length", 128)
    output_dir = cfg.get("output_dir", "./rlaif_output")
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
            "prompt": ["What is 2+2?", "Write a function.", "Explain math"]
        })
    else:
        dataset_path = cfg.get("dataset_path")
        if not dataset_path:
            raise ValueError("dataset_path must be provided in config for RLAIF")
        dataset = load_dataset(dataset_path, split="train")

    prompts = dataset["prompt"]

    print(f"Generating 2 responses per prompt for {len(prompts)} prompts...")
    all_completions = generate_responses(model, tokenizer, prompts, 2, max_completion_length)

    rlaif_data = []
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

        rlaif_data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    # Save to jsonl
    os.makedirs(output_dir, exist_ok=True)
    generated_data_path = os.path.join(output_dir, "rlaif_dataset.jsonl")
    with open(generated_data_path, "w", encoding="utf-8") as f:
        for item in rlaif_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(rlaif_data)} RLAIF instances to {generated_data_path}")
    return generated_data_path
