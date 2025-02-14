from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path
from dotenv import load_dotenv
@dataclass
class Config:
    """Configuration settings for the Assessment Agent."""
    load_dotenv()

    # API Settings
    llm_api_url: str = ""
    llm_model_name: Optional[str] = ""
    
    # Database Settings
    vector_db_path: str = "assessment_db"
    
    # Model Settings
    model_name: str = "bert-base-uncased"
    max_sequence_length: int = 512
    embedding_dimension: int = 768
    
    # Text Processing Settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # File Processing Settings
    supported_formats: tuple = ('pdf', 'docx', 'txt', 'html', 'csv')
    input_directory: str = "data"
    
    # Question Generation Settings
    valid_difficulty_levels: tuple = ('easy', 'medium', 'hard')
    valid_question_types: tuple = ('single', 'multiple', 'fill', 'matching', 'free_text')
    questions_per_assessment: int = 10
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        return cls(
            llm_api_url=os.getenv('LLM_API_URL', cls.llm_api_url),
            llm_model_name=os.getenv('LLM_MODEL_NAME', cls.llm_model_name),
            vector_db_path=os.getenv('VECTOR_DB_PATH', cls.vector_db_path),
            model_name=os.getenv('MODEL_NAME', cls.model_name),
            max_sequence_length=int(os.getenv('MAX_SEQUENCE_LENGTH', cls.max_sequence_length)),
            chunk_size=int(os.getenv('CHUNK_SIZE', cls.chunk_size)),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', cls.chunk_overlap)),
        )
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.llm_model_name:
            raise ValueError("LLM Model name is required")
        
        if not Path(self.input_directory).exists():
            Path(self.input_directory).mkdir(parents=True, exist_ok=True)