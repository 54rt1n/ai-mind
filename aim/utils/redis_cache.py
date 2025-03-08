import hashlib
import json
import logging
from typing import Any, Optional, Union

import redis
from ..config import ChatConfig

logger = logging.getLogger(__name__)

class RedisCache:
    """
    Redis cache utility for caching AI-generated summaries and other reusable content.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to avoid multiple Redis connections."""
        if cls._instance is None:
            cls._instance = super(RedisCache, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[ChatConfig] = None):
        """
        Initialize the Redis cache connection.
        
        Args:
            config: ChatConfig instance with Redis connection settings
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self._initialized = True
        self.config = config or ChatConfig.from_env()
        
        # Get Redis configuration from environment or config
        host = getattr(self.config, 'redis_host', 'localhost')
        port = getattr(self.config, 'redis_port', 6379)
        db = getattr(self.config, 'redis_db', 0)
        password = getattr(self.config, 'redis_password', None)
        
        # Cache namespace to avoid key collisions with other Redis users
        self.namespace = getattr(self.config, 'redis_namespace', 'aim:cache:')
        
        # Default expiration for cached items (in seconds)
        # 1 week by default
        self.default_expiry = getattr(self.config, 'redis_cache_expiry', 60 * 60 * 24 * 7)
        
        try:
            self.redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,  # Auto-decode responses to strings
                socket_timeout=5,       # 5-second timeout for operations
                socket_connect_timeout=5,
                health_check_interval=30
            )
            self.redis.ping()  # Test connection
            logger.info(f"Connected to Redis cache at {host}:{port}/{db}")
        except redis.ConnectionError as e:
            logger.warning(f"Failed to connect to Redis cache: {e}")
            self.redis = None
            
    def _make_key(self, key: str) -> str:
        """
        Create a namespaced key to avoid collisions.
        
        Args:
            key: The base key
            
        Returns:
            str: Namespaced key
        """
        return f"{self.namespace}{key}"
        
    def _hash_content(self, content: str, parameters: Optional[dict] = None) -> str:
        """
        Create a hash from content and optional parameters.
        
        Args:
            content: The content to hash
            parameters: Optional parameters that affect the result
            
        Returns:
            str: MD5 hash of the content and parameters
        """
        # Create a stable representation of the content and parameters
        hash_input = content
        
        if parameters:
            # Sort parameters by key for stable serialization
            sorted_params = {k: parameters[k] for k in sorted(parameters.keys())}
            hash_input += json.dumps(sorted_params)
            
        # Create MD5 hash of the combined input
        return hashlib.md5(hash_input.encode('utf-8')).hexdigest()
        
    def get(self, key: str) -> Optional[str]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[str]: The cached value or None if not found
        """
        if not self.redis:
            return None
            
        try:
            return self.redis.get(self._make_key(key))
        except Exception as e:
            logger.warning(f"Redis cache get error: {e}")
            return None
            
    def set(self, key: str, value: str, expire: Optional[int] = None) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            expire: Expiration time in seconds (uses default if None)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.redis:
            return False
            
        try:
            return self.redis.set(
                self._make_key(key),
                value,
                ex=expire if expire is not None else self.default_expiry
            )
        except Exception as e:
            logger.warning(f"Redis cache set error: {e}")
            return False
            
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.redis:
            return False
            
        try:
            return bool(self.redis.delete(self._make_key(key)))
        except Exception as e:
            logger.warning(f"Redis cache delete error: {e}")
            return False
            
    def get_or_cache(self, content: str, generator_func, parameters: Optional[dict] = None, expire: Optional[int] = None) -> Any:
        """
        Get a result from cache or generate and cache it if not present.
        
        Args:
            content: The input content
            generator_func: Function to call if cache miss (will receive content and parameters)
            parameters: Optional parameters to pass to the generator function
            expire: Cache expiration time in seconds
            
        Returns:
            Any: The cached or newly generated result
        """
        if not self.redis:
            # If Redis is not available, just generate the result
            return generator_func(content, **(parameters or {}))
            
        # Create a hash from the content and parameters
        content_hash = self._hash_content(content, parameters)
        
        # Try to get from cache
        cached_result = self.get(content_hash)
        if cached_result:
            logger.info(f"Cache hit for content hash {content_hash[:8]}...")
            return cached_result
            
        # Generate new result
        logger.info(f"Cache miss for content hash {content_hash[:8]}...")
        result = generator_func(content, **(parameters or {}))
        
        # Cache the result
        if result:
            self.set(content_hash, result, expire)
            
        return result 