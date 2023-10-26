import openai

def gpt_paraphrase(original_text, prompt=None, paraphrase_model_name=None, paraphrase_temperature=None, paraphrase_max_tokens=None):
    assert prompt, "Prompt must be provided for GPT attack"

    paraphrase_query = prompt + original_text
    query_msg = {"role": "user", "content": paraphrase_query}

    from tenacity import retry, stop_after_attempt, wait_random_exponential

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(25))
    def completion_with_backoff(model, messages, temperature, max_tokens):
        return openai.ChatCompletion.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )

    outputs = completion_with_backoff(
        model=paraphrase_model_name,
        messages=[query_msg],
        temperature=paraphrase_temperature,
        max_tokens=paraphrase_max_tokens,
    )

    paraphrase_prompt = outputs.choices[0].message.content

    return paraphrase_prompt