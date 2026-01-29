# aim/conversation/embedding.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Optional


class HuggingFaceEmbedding:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        """
        HuggingFaceEmbedding class for generating embeddings using Hugging Face models.

        Args:
            model_name (str): The name of the pre-trained model to use.
            device (str): The device to use for computation (e.g., "cpu", "cuda:0", etc.).

        Usage:
            embedding = HuggingFaceEmbedding()
            text_embedding_vector = embedding("This is a sample text.")
        """
        self.model_name = model_name
        # Initialize the tokenizer and model
        # the clean_up_tokenization_spaces is explicitly set to the default to suppress a warning
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        if device is not None:
            self.work_device = device
        else:
            self.work_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __call__(self, text: str) -> np.ndarray:
        """
        Calculates the embedding for the given text using the pre-trained model.
        
        Args:
            text (str): The input text to calculate the embedding for.
        
        Returns:
            np.ndarray: The embedding vector for the input text.
        """
        
        if self.work_device != "cpu":
            self.model.to(self.work_device)
        
        result = self._get_embedding(text)
        
        if self.work_device != "cpu":
            self.model.to('cpu')
        
        return result

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Calculates the embedding for the given text using the pre-trained model.
        
        Args:
            text (str): The input text to calculate the embedding for.
        
        Returns:
            np.ndarray: The embedding vector for the input text.
        """

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        if self.work_device != "cpu":
            inputs = inputs.to(self.work_device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings: np.ndarray = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
        
        return embeddings[0]

    def transform(self, texts: list[str]) -> list[np.ndarray]:
        """
        Transforms a list of texts into a list of embeddings.
        """

        if self.work_device != "cpu":
            self.model.to(self.work_device)

        results = [self._get_embedding(text) for text in texts]

        if self.work_device != "cpu":
            self.model.to('cpu')

        return results

    def to_device(self) -> None:
        """Move model to work device (GPU) for batch processing."""
        if self.work_device != "cpu":
            self.model.to(self.work_device)

    def to_cpu(self) -> None:
        """Move model back to CPU after batch processing."""
        if self.work_device != "cpu":
            self.model.to('cpu')

    def transform_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Transform texts without moving model (assumes model already on device).
        Use with to_device()/to_cpu() for efficient batch processing.
        """
        return [self._get_embedding(text) for text in texts]