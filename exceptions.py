class AssessmentAgentError(Exception):
    """Base exception class for Assessment Agent."""
    pass

class FileProcessingError(AssessmentAgentError):
    """Raised when there's an error processing input files."""
    pass

class EmbeddingError(AssessmentAgentError):
    """Raised when there's an error generating embeddings."""
    pass

class DatabaseError(AssessmentAgentError):
    """Raised when there's an error with the vector database."""
    pass

class LLMAPIError(AssessmentAgentError):
    """Raised when there's an error communicating with the LLM API."""
    pass

class ValidationError(AssessmentAgentError):
    """Raised when there's an error with input validation."""
    pass