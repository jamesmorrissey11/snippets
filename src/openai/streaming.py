import openai


def stream_response(user_message, model_name):
    collected_chunks = []
    collected_messages = []
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": user_message}],
        temperature=0,
        stream=True,
    )
    for i, chunk in enumerate(response):
        collected_chunks.append(chunk)
        chunk_message = chunk["choices"][0]["delta"]
        collected_messages.append(chunk_message)
