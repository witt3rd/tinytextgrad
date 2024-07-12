from litellm import completion


def call_llm(
    prompt: str,
    prompt_input: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.9,
    max_tokens: int = 4096,
    top_p: float = 0.95,
    frequency_penalty: float = 0,
    format_as_json: bool = False,
) -> str:
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
        response_format={"type": "json_object" if format_as_json else "text"},
    )
    return response.choices[0].message.content
