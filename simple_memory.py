import os
import time
import numpy as np
from datetime import datetime
from typing import List, Literal, Optional
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv

# Load env to get OPENAI_API_KEY
load_dotenv()

@dataclass
class MemoryItem:
    content: str
    memory_type: Literal['episodic', 'semantic', 'procedural']
    vector: np.ndarray = field(repr=False)
    timestamp: float = field(default_factory=time.time)
    
    def get_age_hours(self) -> float:
        return (time.time() - self.timestamp) / 3600.0

class LightMemoryStore:
    def __init__(self, decay_rate: float = 0.01):
        """
        :param decay_rate: How fast memories fade per hour. 
                           0.01 = ~1% signal loss per hour.
        """
        self.client = OpenAI()
        self.memories: List[MemoryItem] = []
        self.decay_rate = decay_rate
        print("🧠 LightMemoryStore initialized.")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Fetch embedding from OpenAI (1536 dims)."""
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return np.array(response.data[0].embedding)

    def add(self, content: str, memory_type: Literal['episodic', 'semantic', 'procedural']):
        """Write a new memory."""
        vector = self._get_embedding(content)
        memory = MemoryItem(content=content, memory_type=memory_type, vector=vector)
        self.memories.append(memory)
        print(f"📝 Added [{memory_type}]: '{content}'")

    def retrieve(self, query: str, top_k: int = 3) -> List[tuple]:
        """
        Retrieve memories based on:
        Score = (Vector_Similarity * 0.7) + (Recency_Score * 0.3)
        Recency_Score = exp(-decay_rate * age_in_hours)
        """
        if not self.memories:
            return []

        query_vec = self._get_embedding(query)
        scored_results = []

        total_weight = 1.0
        sim_weight = 0.7
        recency_weight = 0.3

        print(f"\n🔍 Searching for: '{query}'")
        
        for mem in self.memories:
            # 1. Vector Similarity (Cosine)
            # Dot product is sufficient for normalized vectors (OpenAI returns normalized)
            similarity = np.dot(query_vec, mem.vector)
            
            # 2. Recency Decay (Exponential)
            age = mem.get_age_hours()
            recency_score = np.exp(-self.decay_rate * age)
            
            # 3. Weighted Score
            final_score = (similarity * sim_weight) + (recency_score * recency_weight)
            
            scored_results.append((final_score, mem, similarity, recency_score))

        # Sort by Final Score Descending
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        return scored_results[:top_k]

# --- Demo Usage ---
if __name__ == "__main__":
    store = LightMemoryStore(decay_rate=0.1) # Aggressive decay for demo

    # 1. Add Memories
    store.add("The project code name is Project Chimera.", "semantic")
    store.add("I had a coffee with the team yesterday and we discussed UI.", "episodic")
    store.add("To run the server, use `uvicorn app:main --reload`.", "procedural")
    
    # Simulate time passing (Older memory)
    old_memory = MemoryItem(
        content="Old legacy configuration setting is `use_legacy=True`.",
        memory_type="procedural",
        vector=store._get_embedding("Old legacy configuration setting is `use_legacy=True`."),
        timestamp=time.time() - (24 * 3600) # 24 hours ago
    )
    store.memories.append(old_memory)
    print(f"📝 Added [procedural] (OLD): '{old_memory.content}'")

    # 2. Query
    results = store.retrieve("How do I start the server?")
    
    print(f"\n🏆 Top Results:")
    for score, mem, sim, rec in results:
        print(f"   Score: {score:.4f} (Sim: {sim:.4f}, Recency: {rec:.4f}) | [{mem.memory_type}] {mem.content}")
