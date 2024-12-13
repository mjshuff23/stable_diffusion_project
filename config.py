class Config:
    # Stable Diffusion settings
    SD_MODEL_NAME = "runwayml/stable-diffusion-v1-5"
    TRAINING_EPOCHS = 100
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 4
    
    # Dataset settings
    DATASET_PATH = "images/"
    IMAGE_SIZE = 512
    VALIDATION_SPLIT = 0.1  # 10% of data for validation
    NUM_WORKERS = 4  # For data loading
    
    # LLM settings
    LLM_MODEL_NAME = "Qwen/Qwen1.5-14B"  # or Qwen/Qwen1.5-7B
    MAX_LENGTH = 2048  # Qwen has a longer context window
    TEMPERATURE = 0.7
    TOP_P = 0.9
    USE_FLASH_ATTENTION = True  # Qwen models support flash attention
    TRUST_REMOTE_CODE = True    # Required for Qwen models
    
    # Training settings
    DEVICE = "cuda"  # or "cpu" if no GPU
    SAVE_STEPS = 500
    LOGGING_STEPS = 100
    GRADIENT_ACCUMULATION_STEPS = 1
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    ENABLE_CORS = True
    CORS_ORIGINS = ["http://localhost:3000"]  # Add any frontend URLs
    
    # Paths
    MODEL_SAVE_DIR = "models/"
    LOG_DIR = "logs/"
    CACHE_DIR = ".cache/"

    @classmethod
    def validate_paths(cls):
        """Create necessary directories if they don't exist."""
        import os
        paths = [cls.DATASET_PATH, cls.MODEL_SAVE_DIR, cls.LOG_DIR, cls.CACHE_DIR]
        for path in paths:
            os.makedirs(path, exist_ok=True)