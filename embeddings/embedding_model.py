import time
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME


class EmbeddingModel:
    def __init__(
        self,
        model_name=None,
        local_files_only=False,
    ):
        self.model_name = model_name or EMBEDDING_MODEL_NAME

        start = time.perf_counter()
        print(f"Loading embedding model: {self.model_name}")

        self.model = SentenceTransformer(
            self.model_name,
            local_files_only=local_files_only,
        )

        elapsed = time.perf_counter() - start
        print(f"Model loaded in {elapsed:.2f} seconds.")

    def encode(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text to encode must be a non-empty string.")
        return self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def encode_batch(self, texts):
        if not texts:
            return []
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def get_dimension(self):
        return self.model.get_sentence_embedding_dimension()