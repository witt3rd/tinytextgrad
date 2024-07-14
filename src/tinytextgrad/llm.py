import os

from dotenv import load_dotenv
from litellm import completion

#

load_dotenv()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
DEFAULT_MODEL_TEMPERATURE = float(os.getenv("DEFAULT_MODEL_TEMPERATURE", "0.7"))
DEFAULT_MODEL_MAX_TOKENS = int(os.getenv("DEFAULT_MODEL_MAX_TOKENS", "2048"))
DEFAULT_MODEL_TOP_P = float(os.getenv("DEFAULT_MODEL_TOP_P", "0.95"))
DEFAULT_MODEL_FREQUENCY_PENALTY = float(
    os.getenv("DEFAULT_MODEL_FREQUENCY_PENALTY", "0")
)

#


def call_llm(
    prompt: str,
    prompt_input: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_MODEL_TEMPERATURE,
    max_tokens: int = DEFAULT_MODEL_MAX_TOKENS,
    top_p: float = DEFAULT_MODEL_TOP_P,
    frequency_penalty: float = DEFAULT_MODEL_FREQUENCY_PENALTY,
    response_format_type: str = "text",  # or "json_object"
) -> str:
    """
    Call the LLM with the given prompt and input.
    """
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": prompt_input},
    ]
    response = completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        response_format={"type": response_format_type},
    )
    return response.choices[0].message.content
