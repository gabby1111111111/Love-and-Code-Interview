"""
Core configuration management for AegisIsle
"""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False
            extra = "ignore"
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8001, env="API_PORT")
    api_reload: bool = Field(default=True, env="API_RELOAD")

    # Database Configuration
    database_url: str = Field(default="sqlite:///./aegis_isle.db", env="DATABASE_URL")
    vector_db_type: str = Field(default="qdrant", env="VECTOR_DB_TYPE")
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_collection: str = Field(default="aegis_isle_documents", env="QDRANT_COLLECTION")

    # LLM Configuration
    llm_provider: str = Field(default="openai", env="LLM_PROVIDER")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    openai_base_url: Optional[str] = Field(default=None, env="OPENAI_BASE_URL")
    # Model Settings
    default_llm_model: str = Field(default="gpt-4-1106-preview", env="DEFAULT_LLM_MODEL")
    embedding_model: str = Field(default="text-embedding-ada-002", env="EMBEDDING_MODEL")
    max_tokens: int = Field(default=4096, env="MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")

    # RAG Configuration
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    max_retrieved_docs: int = Field(default=5, env="MAX_RETRIEVED_DOCS")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")

    # Agent Configuration
    max_agent_iterations: int = Field(default=10, env="MAX_AGENT_ITERATIONS")
    agent_timeout: int = Field(default=300, env="AGENT_TIMEOUT")
    enable_memory: bool = Field(default=True, env="ENABLE_MEMORY")

    # File Processing
    upload_max_size: str = Field(default="50MB", env="UPLOAD_MAX_SIZE")
    supported_formats: str = Field(default="pdf,docx,txt,md,html", env="SUPPORTED_FORMATS")
    ocr_enabled: bool = Field(default=True, env="OCR_ENABLED")
    ocr_language: str = Field(default="eng+chi_sim", env="OCR_LANGUAGE")

    # Security & Authentication
    secret_key: str = Field(default="change-this-in-production", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    allowed_hosts: str = Field(default="localhost,127.0.0.1", env="ALLOWED_HOSTS")

    # OAuth2 + RBAC Configuration
    admin_username: str = Field(default="admin", env="ADMIN_USERNAME")
    admin_password: str = Field(default="admin123", env="ADMIN_PASSWORD")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")

    # Audit Logging Configuration
    audit_log_enabled: bool = Field(default=True, env="AUDIT_LOG_ENABLED")
    audit_log_retention_days: int = Field(default=365, env="AUDIT_LOG_RETENTION_DAYS")
    structured_logging: bool = Field(default=True, env="STRUCTURED_LOGGING")
    elk_compatible: bool = Field(default=True, env="ELK_COMPATIBLE")

    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    log_requests: bool = Field(default=True, env="LOG_REQUESTS")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")

    # Multi-modal
    enable_multimodal: bool = Field(default=True, env="ENABLE_MULTIMODAL")
    image_processing_enabled: bool = Field(default=True, env="IMAGE_PROCESSING_ENABLED")
    vision_model: str = Field(default="gpt-4-vision-preview", env="VISION_MODEL")

    # Computed properties
    @property
    def supported_formats_list(self) -> List[str]:
        """Get supported file formats as a list."""
        return [fmt.strip() for fmt in self.supported_formats.split(",")]

    @property
    def allowed_hosts_list(self) -> List[str]:
        """Get allowed hosts as a list."""
        return [host.strip() for host in self.allowed_hosts.split(",")]

    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent.parent.parent

    @property
    def data_dir(self) -> Path:
        """Get data directory."""
        return self.project_root / "data"

    @property
    def uploads_dir(self) -> Path:
        """Get uploads directory."""
        return self.data_dir / "uploads"

    @property
    def models_dir(self) -> Path:
        """Get models directory."""
        return self.project_root / "models"

    @property
    def config_dir(self) -> Path:
        """Get config directory."""
        return self.project_root / "config"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()