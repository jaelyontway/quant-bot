"""Text embeddings"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional

DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
_global_model: Optional[SentenceTransformer] = None

def embed_texts(texts: List[str], model_name: str = DEFAULT_MODEL, batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
    global _global_model
    if _global_model is None or _global_model.model_name != model_name:
        _global_model = SentenceTransformer(model_name)
        _global_model.model_name = model_name
    embeddings = _global_model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress, convert_to_numpy=True)
    return embeddings

