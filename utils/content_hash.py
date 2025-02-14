import hashlib
from pathlib import Path
from typing import Optional
import os

def calculate_file_hash(file_path: str) -> str:
    """
    Calculate SHA-256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: Hex digest of file hash
    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Read the file in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
            
    return sha256_hash.hexdigest()

def get_file_metadata(file_path: str) -> dict:
    """
    Get file metadata including hash, size, and modification time.
    
    Args:
        file_path: Path to the file
        
    Returns:
        dict: File metadata
    """
    path = Path(file_path)
    stat = path.stat()
    
    return {
        "file_path": str(path),
        "file_name": path.name,
        "file_hash": calculate_file_hash(file_path),
        "file_size": stat.st_size,
        "modified_time": stat.st_mtime,
    }