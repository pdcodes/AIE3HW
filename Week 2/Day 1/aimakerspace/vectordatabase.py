import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Callable
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio

def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)

class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.vectors = defaultdict(np.array)
        self.metadata = {}  # Dictionary to store metadata
        self.embedding_model = embedding_model or EmbeddingModel()

    def insert(self, key: str, vector: np.array, metadata: Dict[str, str] = None) -> None:
        self.vectors[key] = vector
        if metadata:
            self.metadata[key] = metadata

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_metadata: bool = False
    ) -> List[Tuple[str, float]]:
        scores = [
            (key, distance_measure(query_vector, vector))
            for key, vector in self.vectors.items()
        ]
        top_k = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
        if return_metadata:
            return [(key, score, self.metadata.get(key, {})) for key, score in top_k]
        return top_k

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
        return_metadata: bool = False
    ) -> List[Tuple[str, float]]:
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure, return_metadata)
        if return_as_text:
            return [result[0] for result in results]
        return results

    def retrieve_from_key(self, key: str) -> Tuple[np.array, Dict[str, str]]:
        return self.vectors.get(key, None), self.metadata.get(key, None)

    async def abuild_from_list(self, list_of_text: List[str], list_of_topics: List[str]) -> "VectorDatabase":
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for text, topics, embedding in zip(list_of_text, list_of_topics, embeddings):
            self.insert(text, np.array(embedding), {'main_topics': topics})
        return self

if __name__ == "__main__":
    # Example usage
    list_of_text = ["Sample text 1", "Sample text 2"]
    list_of_topics = ["Topic 1, Topic 2", "Topic 3, Topic 4"]
    
    db = VectorDatabase()
    asyncio.run(db.abuild_from_list(list_of_text, list_of_topics))
    
    # Searching with metadata
    query = "Sample query text"
    results = db.search_by_text(query, k=2, return_metadata=True)
    print(results)

'''
if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text))
    k = 2

    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    retrieved_vector = vector_db.retrieve_from_key(
        "I like to eat broccoli and bananas."
    )
    print("Retrieved vector:", retrieved_vector)

    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)
'''