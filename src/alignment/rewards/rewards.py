from typing import List, Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


def accuracy_reward(completions: List[str]) -> List[float]:
    rewards = []
    for text in completions:
        if "<solution>" in text:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def format_reward(completions: List[str]) -> List[float]:
    """Checks if the response follows a specific format (e.g. starts with a particular tag or contains code blocks)"""
    rewards = []
    for text in completions:
        if "```" in text:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def length_penalty_reward(completions: List[str]) -> List[float]:
    """Penalize extremely long responses to prevent reward hacking"""
    rewards = []
    for text in completions:
        length = len(text)
        if length > 1000:
            rewards.append(-0.5)
        elif length > 500:
            rewards.append(0.0)
        else:
            rewards.append(0.5)
    return rewards


def diffusion_trajectory_reward(completions: List[str]) -> List[float]:
    """Simulates evaluating parallel unmasking states of a diffusion model."""
    rewards = []
    for text in completions:
        if len(text.strip()) > 0:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards

class ModelBasedReward:
    def __init__(self, config: Dict[str, Any]):
        self.api_type = config.get("api_type", "transformers")
        self.model_name = config.get("model_name", "gpt-3.5-turbo")
        self.system_prompt = config.get("system_prompt", "Evaluate the following response to the prompt. Return a JSON object with a single key 'score' containing a float between 0.0 and 1.0.")

        if self.api_type == "openai":
            from openai import OpenAI
            api_key = config.get("api_key")
            base_url = config.get("base_url")

            client_kwargs = {}
            if api_key:
                client_kwargs["api_key"] = api_key
            if base_url:
                client_kwargs["base_url"] = base_url

            self.client = OpenAI(**client_kwargs)
        elif self.api_type == "transformers":
            from transformers import pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device_map="auto"
            )
        else:
            raise ValueError(f"Unknown api_type for ModelBasedReward: {self.api_type}")

    def __call__(self, completions: List[str], prompts: Optional[List[str]] = None) -> List[float]:
        if prompts is None:
            # Fallback if prompts are not provided
            prompts = [""] * len(completions)

        rewards = []
        for prompt, completion in zip(prompts, completions):
            input_text = f"Prompt: {prompt}\n\nResponse: {completion}"

            if self.api_type == "openai":
                try:
                    schema = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "reward_score",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "score": {
                                        "type": "number",
                                        "description": "A score between 0.0 and 1.0 evaluating the response"
                                    }
                                },
                                "required": ["score"],
                                "additionalProperties": False
                            }
                        }
                    }

                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": input_text}
                        ],
                        response_format=schema,
                        temperature=0.0
                    )
                    content = response.choices[0].message.content
                    parsed = json.loads(content)
                    score = float(parsed.get("score", 0.0))
                    rewards.append(score)
                except Exception as e:
                    logger.error(f"OpenAI API error in ModelBasedReward: {e}")
                    rewards.append(0.0)

            elif self.api_type == "transformers":
                try:
                    # Instruct the local model to generate valid JSON
                    full_prompt = f"{self.system_prompt}\n\n{input_text}\n\nJSON output:\n{{"
                    outputs = self.pipeline(
                        full_prompt,
                        max_new_tokens=20,
                        return_full_text=False,
                        pad_token_id=self.pipeline.tokenizer.eos_token_id if self.pipeline.tokenizer else None
                    )
                    generated_text = outputs[0]["generated_text"].strip()

                    # Try to parse the generated text as JSON, assuming it starts after the '{' we provided
                    try:
                        # Simple extraction strategy: find first number
                        import re
                        match = re.search(r'[-+]?\d*\.\d+|\d+', generated_text)
                        if match:
                            score = float(match.group())
                            rewards.append(score)
                        else:
                            rewards.append(0.0)
                    except Exception:
                        rewards.append(0.0)
                except Exception as e:
                    logger.error(f"Transformers pipeline error in ModelBasedReward: {e}")
                    rewards.append(0.0)

        return rewards


def get_reward_function(name: str, config: Optional[Dict[str, Any]] = None):
    if name == "accuracy":
        return accuracy_reward
    elif name == "format":
        return format_reward
    elif name == "length_penalty":
        return length_penalty_reward
    elif name == "diffusion_trajectory":
        return diffusion_trajectory_reward
    elif name == "model_based":
        if config is None:
            config = {}
        return ModelBasedReward(config)
    else:
        raise ValueError(f"Unknown reward function: {name}")
