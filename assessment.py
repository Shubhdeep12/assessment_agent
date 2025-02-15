import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests
from transformers import AutoTokenizer, AutoModel
import torch
import pypdf
import docx
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import lancedb
from lancedb.pydantic import Vector, LanceModel

from config import Config
from exceptions import (
    AssessmentAgentError, FileProcessingError, EmbeddingError,
    DatabaseError, LLMAPIError, ValidationError
)
from utils.rate_limiter import RateLimiter
from utils.content_hash import get_file_metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('assessment_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define table schemas using Pydantic models
class ContentSchema(LanceModel):
    id: str
    text: str
    vector: Vector(768)  # BERT embedding dimension
    file_hash: str

class MetadataSchema(LanceModel):
    file_path: str
    file_name: str
    file_hash: str
    file_size: int
    modified_time: float
    processed_time: float

class AssessmentAgent:
    def __init__(self, config: Config):
        """
        Initialize the assessment agent with necessary components.
        
        Args:
            config: Configuration object containing all settings
        """
        self.config = config
        self.config.validate()
        
        # Initialize components
        self._init_model()
        self._init_database()
        self._init_rate_limiter()
        
    def _init_model(self) -> None:
        """Initialize the embedding model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModel.from_pretrained(self.config.model_name)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize model: {e}")
            
    def _init_database(self) -> None:
        """Initialize the vector database."""
        try:
            self.db = lancedb.connect(self.config.vector_db_path)
            self.table = self._initialize_table()
            self.metadata_table = self._initialize_metadata_table()
        except Exception as e:
            raise DatabaseError(f"Failed to initialize database: {e}")
            
    def _init_rate_limiter(self) -> None:
        """Initialize rate limiter for LLM API."""
        self.rate_limiter = RateLimiter(
            tokens=100,  # Adjust based on API limits
            refill_time=60.0  # Tokens refill per minute
        )

    def _initialize_table(self) -> lancedb.table.Table:
        """Initialize or get the vector database table."""
        try:
            if "content" not in self.db.table_names():
                # Create initial data
                initial_data = ContentSchema(
                    id="init",
                    text="initialization",
                    vector=np.zeros(self.config.embedding_dimension),
                    file_hash="init"
                )
                
                table = self.db.create_table(
                    "content",
                    data=[initial_data.dict()],
                    schema=ContentSchema
                )
            else:
                table = self.db.open_table("content")
            return table
        except Exception as e:
            raise DatabaseError(f"Failed to initialize content table: {e}")

    def _initialize_metadata_table(self) -> lancedb.table.Table:
        """Initialize or get the metadata table."""
        try:
            if "metadata" not in self.db.table_names():
                # Create initial data
                initial_data = MetadataSchema(
                    file_path="init",
                    file_name="init",
                    file_hash="init",
                    file_size=0,
                    modified_time=0.0,
                    processed_time=0.0
                )
                
                table = self.db.create_table(
                    "metadata",
                    data=[initial_data.dict()],
                    schema=MetadataSchema
                )
            else:
                table = self.db.open_table("metadata")
            return table
        except Exception as e:
            raise DatabaseError(f"Failed to initialize metadata table: {e}")

    def _check_file_processed(self, file_metadata: Dict[str, Any]) -> bool:
        """
        Check if file has already been processed by comparing metadata.
        
        Args:
            file_metadata: File metadata including hash
            
        Returns:
            bool: True if file has been processed and cached
        """
        try:
            # Search for file in metadata table
            results = self.metadata_table.search().where(
                f"file_hash = '{file_metadata['file_hash']}'"
            ).to_list()
            
            if not results:
                return False
                
            # Compare metadata to check if file has changed
            stored_metadata = results[0]
            return (
                stored_metadata["file_size"] == file_metadata["file_size"] and
                stored_metadata["modified_time"] == file_metadata["modified_time"]
            )
        except Exception as e:
            logger.warning(f"Error checking file cache: {e}")
            return False

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from various file formats.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            str: Extracted text content
            
        Raises:
            FileProcessingError: If file processing fails
        """
        try:
            file_extension = Path(file_path).suffix.lower()[1:]
            
            if file_extension not in self.config.supported_formats:
                raise ValidationError(f"Unsupported file format: {file_extension}")
            
            if file_extension == 'pdf':
                text = self._extract_from_pdf(file_path)
            elif file_extension == 'docx':
                text = self._extract_from_docx(file_path)
            elif file_extension == 'txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            elif file_extension == 'html':
                with open(file_path, 'r', encoding='utf-8') as file:
                    soup = BeautifulSoup(file.read(), 'html.parser')
                    text = soup.get_text(separator='\n')
            elif file_extension == 'csv':
                df = pd.read_csv(file_path)
                text = df.to_string()
            
            logger.info(f"Successfully extracted text from {file_path}")
            return text
            
        except Exception as e:
            raise FileProcessingError(f"Failed to extract text from {file_path}: {e}")

    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using PyPDF."""
        text_content = []
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text_content.append(page.extract_text())
        return '\n'.join(text_content)

    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX using python-docx."""
        doc = docx.Document(file_path)
        text_content = []
        for paragraph in doc.paragraphs:
            text_content.append(paragraph.text)
        return '\n'.join(text_content)

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks while preserving structure."""
        try:
            chunks = []
            sentences = text.split('.')
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip() + '.'
                sentence_length = len(sentence)
                
                if current_length + sentence_length > chunk_size and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)
                    
                    # Calculate overlap
                    overlap_words = chunk_text.split()[-overlap:]
                    current_chunk = [' '.join(overlap_words), sentence]
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                
            logger.info(f"Text split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            raise FileProcessingError(f"Failed to chunk text: {e}")

    def generate_embeddings(self, chunks: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings using the model.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List[np.ndarray]: List of embeddings
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            embeddings = []
            for chunk in chunks:
                inputs = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_sequence_length,
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(embedding[0])
                
            logger.info(f"Generated embeddings for {len(chunks)} chunks")
            return embeddings
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {e}")

    def store_chunks(
        self,
        chunks: List[str],
        embeddings: List[np.ndarray],
        file_hash: str
    ) -> None:
        """
        Store text chunks and their embeddings in the vector database.
        
        Args:
            chunks: List of text chunks
            embeddings: List of corresponding embeddings
            file_hash: Hash of the source file
            
        Raises:
            DatabaseError: If storing in database fails
        """
        try:
            data = [
                ContentSchema(
                    id=f"{file_hash}_{i}",
                    text=chunk,
                    vector=embedding,
                    file_hash=file_hash
                ).dict()
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ]
            
            self.table.add(data)
            logger.info(f"Stored {len(chunks)} chunks in the database")
        except Exception as e:
            raise DatabaseError(f"Failed to store chunks in database: {e}")

    def store_file_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Store file metadata in the metadata table.
        
        Args:
            metadata: File metadata
            
        Raises:
            DatabaseError: If storing metadata fails
        """
        try:
            metadata_record = MetadataSchema(
                **metadata,
                processed_time=time.time()
            )
            self.metadata_table.add([metadata_record.dict()])
            logger.info(f"Stored metadata for file: {metadata['file_path']}")
        except Exception as e:
            raise DatabaseError(f"Failed to store file metadata: {e}")

    def search_similar_chunks(
        self,
        query_embedding: np.ndarray,
        limit: int = 5
    ) -> List[str]:
        """
        Retrieve similar text chunks from the vector database.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results to return
            
        Returns:
            List[str]: List of similar text chunks
            
        Raises:
            DatabaseError: If search fails
        """
        try:
            results = self.table.search(query_embedding).limit(limit).to_list()
            return [result['text'] for result in results]
        except Exception as e:
            raise DatabaseError(f"Failed to search similar chunks: {e}")

    def generate_questions(
        self,
        context: str,
        difficulty: str,
        question_type: str
    ) -> str:
        """
        Generate assessment questions using LLM API.
        
        Args:
            context: Content to generate questions from
            difficulty: Difficulty level of questions
            question_type: Type of questions to generate
            
        Returns:
            str: Generated questions
            
        Raises:
            ValidationError: If input parameters are invalid
            LLMAPIError: If API request fails
        """
        # Validate inputs
        if difficulty not in self.config.valid_difficulty_levels:
            raise ValidationError(f"Invalid difficulty level: {difficulty}")
        if question_type not in self.config.valid_question_types:
            raise ValidationError(f"Invalid question type: {question_type}")
            
        try:
            # Wait for rate limit
            if not self.rate_limiter.acquire(timeout=30.0):
                raise LLMAPIError("Rate limit exceeded")
            
            prompt = f"""Based on the following content, generate {self.config.questions_per_assessment} {difficulty} level {question_type} questions:

Content:
{context}

Requirements:
- Generate exactly {self.config.questions_per_assessment} questions
- Difficulty level: {difficulty}
- Question type: {question_type}
- Include correct answers
- For single choice and multiple choice questions, include 4 options
- Ensure questions test understanding, not just memorization
"""
            # Prepare headers
            headers = {
                'Content-Type': 'application/json',
            }

            # Prepare payload
            prompt = prompt.replace('\n', ' ')
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an instructor responsible for creating assessments from lesson content"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False,
                "model": self.config.llm_model_name
            }

            # Make request
            response = requests.post(
                f"{self.config.llm_api_url}/api/chat",
                json=payload,
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('message', {}).get('content', '')
        
        except requests.exceptions.RequestException as e:
            raise LLMAPIError(f"Failed to communicate with LLM API: {e}")
        except Exception as e:
            raise LLMAPIError(f"Failed to generate questions: {e}")

    def run_chat_interface(self) -> None:
        """Run the terminal-based chat interface."""
        while True:
            try:
                print("\n=== AI Assessment Agent ===")
                
                # Get difficulty level
                difficulty = input(
                    f"Enter difficulty level ({'/'.join(self.config.valid_difficulty_levels)}): "
                ).lower()
                if difficulty not in self.config.valid_difficulty_levels:
                    print("Invalid difficulty level. Please try again.")
                    continue

                # Get question type
                question_type = input(
                    f"Enter question type ({'/'.join(self.config.valid_question_types)}): "
                ).lower()
                if question_type not in self.config.valid_question_types:
                    print("Invalid question type. Please try again.")
                    continue

                # Create query embedding
                query = f"Generate {difficulty} {question_type} questions"
                inputs = self.tokenizer(
                    query,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_sequence_length
                )
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    query_embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]

                # Retrieve and generate
                similar_chunks = self.search_similar_chunks(query_embedding)
                context = " ".join(similar_chunks)
                questions = self.generate_questions(context, difficulty, question_type)
                
                print("\nGenerated Questions:")
                print(questions)

                if input("\nGenerate more questions? (y/n): ").lower() != 'y':
                    break

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error in chat interface: {e}")
                print("An error occurred. Please try again.")
                break

def process_file(file_path: str, agent: AssessmentAgent) -> None:
    """
    Process a single file and store its content in the vector database.
    
    Args:
        file_path: Path to the file to process
        agent: AssessmentAgent instance
    """
    try:
        # Get file metadata
        metadata = get_file_metadata(file_path)
        
        # Check if file has already been processed
        if agent._check_file_processed(metadata):
            logger.info(f"File {file_path} already processed, using cached content")
            return
            
        # Process new file
        text = agent.extract_text(file_path)
        chunks = agent.chunk_text(text)
        embeddings = agent.generate_embeddings(chunks)
        
        # Store chunks with file hash
        agent.store_chunks(chunks, embeddings, metadata["file_hash"])
        
        # Store metadata
        agent.store_file_metadata(metadata)
        
        logger.info(f"Successfully processed file: {file_path}")
    except AssessmentAgentError as e:
        logger.error(f"Failed to process file {file_path}: {e}")
        raise

def main() -> None:
    """Main function to run the assessment agent."""
    try:
        # Load configuration
        config = Config.from_env()
        
        # Initialize agent
        agent = AssessmentAgent(config)

        # Process input files
        input_dir = Path(config.input_directory)
        if not input_dir.exists():
            input_dir.mkdir(parents=True)
            logger.info(f"Created input directory: {input_dir}")
            print(f"Please place your files in the '{input_dir}' directory and run the script again.")
            return

        files = list(input_dir.glob("*.*"))
        if not files:
            print(f"No files found in '{input_dir}'. Please add some files and try again.")
            return

        # Process each file
        for file_path in files:
            process_file(str(file_path), agent)

        # Start the chat interface
        agent.run_chat_interface()

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print("An error occurred. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()