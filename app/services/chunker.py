import tiktoken
from typing import List
from app.config import settings

class DocumentChunker:
    """Chunks documents into fixed-size overlapping windows"""
    
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
    
    def chunk(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks based on token count.
        
        Returns:
            List of text chunks
        """
        # Encode to tokens
        tokens = self.encoding.encode(text)
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            # Extract chunk
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            
            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move forward with overlap
            if end >= len(tokens):
                break
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
