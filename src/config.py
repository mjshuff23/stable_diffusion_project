class Config:
    # Stable Diffusion settings
    SD_MODEL_NAME = "runwayml/stable-diffusion-v1-5" # or "runwayml/stable-diffusion-v1-5-tiny"
    TRAINING_EPOCHS = 100 # epochs are rounds of training
    LEARNING_RATE = 1e-5 # 1e-5 is a good starting point
    BATCH_SIZE = 4 # 4 is a good starting point
    
    # Dataset settings
    DATASET_PATH = "images/" # Path to the dataset
    IMAGE_SIZE = 512 # Images will be resized to this size
    VALIDATION_SPLIT = 0.1  # 10% of data for validation, meaning 90% for training
    NUM_WORKERS = 4  # For data loading
    
    # LLM settings
    LLM_MODEL_NAME = "Qwen/Qwen1.5-14B"  # or Qwen/Qwen1.5-7B
    MAX_LENGTH = 2048  # Qwen has a longer context window
    TEMPERATURE = 0.7 # Temperature affects the randomness of the generated text
    TOP_P = 0.9 # TOP_P affects the diversity of the generated text
    USE_FLASH_ATTENTION = True  # Qwen models support flash attention, which is faster
    TRUST_REMOTE_CODE = True    # Required for Qwen models
    
    # Training settings
    DEVICE = "cuda"  # or "cpu" if no GPU is available
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