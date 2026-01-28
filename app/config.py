from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    database_url: str
    
    # OpenAI
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    llm_model: str = "gpt-4o-mini"
    
    # Chunking
    chunk_size: int = 400
    chunk_overlap: int = 50
    
    # Retrieval
    top_k: int = 20
    alpha_semantic: float = 0.6
    beta_recency: float = 0.3
    gamma_graph: float = 0.1
    token_budget: int = 4000
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
