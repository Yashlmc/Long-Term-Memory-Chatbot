# pyrefly: ignore [missing-import]
from openrouter import OpenRouter

from config import MODEL_NAME, OPENROUTER_API_KEY


class OpenRouterClient:
    def __init__(self, api_key=None, model_name=None):
        key = api_key or OPENROUTER_API_KEY
        if not key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set in the .env file."
            )

        self.model_name = model_name or MODEL_NAME
        self.client = OpenRouter(api_key=key)

    def generate_response(self, messages, temperature=0.7):
        response = self.client.chat.send(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )

        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError("OpenRouter returned no response choices.")

        content = getattr(choices[0].message, "content", None)
        if content is None:
            raise RuntimeError("OpenRouter returned an empty message content.")

        return content
