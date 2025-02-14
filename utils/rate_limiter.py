from typing import Dict, Optional
import time
from threading import Lock

class RateLimiter:
    """Simple token bucket rate limiter."""
    
    def __init__(self, tokens: int, refill_time: float):
        """
        Initialize rate limiter.
        
        Args:
            tokens: Maximum number of tokens in the bucket
            refill_time: Time in seconds between token refills
        """
        self.tokens = tokens
        self.refill_time = refill_time
        self.current_tokens = tokens
        self.last_update = time.time()
        self.lock = Lock()
        
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        new_tokens = int(elapsed / self.refill_time * self.tokens)
        
        if new_tokens > 0:
            self.current_tokens = min(self.tokens, self.current_tokens + new_tokens)
            self.last_update = now
    
    def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Attempt to acquire tokens.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait for tokens
            
        Returns:
            bool: True if tokens were acquired, False otherwise
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                self._refill()
                
                if self.current_tokens >= tokens:
                    self.current_tokens -= tokens
                    return True
                
            if timeout is not None and time.time() - start_time > timeout:
                return False
                
            time.sleep(0.1)  # Prevent busy waiting