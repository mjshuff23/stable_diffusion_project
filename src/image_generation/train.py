from diffusers import StableDiffusionPipeline
import torch
from datasets import load_dataset
from ..config import Config
from transformers import TrainingArguments, Trainer

def train_stable_diffusion():
    """Fine-tune Stable Diffusion on custom dataset."""
    # Load model
    pipeline = StableDiffusionPipeline.from_pretrained(
        Config.SD_MODEL_NAME,
        torch_dtype=torch.float16
    )
    
    # Load and prepare dataset
    dataset = load_dataset("imagefolder", data_dir=Config.DATASET_PATH)
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir="./stable-diffusion-finetuned",
        learning_rate=Config.LEARNING_RATE,
        num_train_epochs=Config.TRAINING_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
    )
    
    # Start training
    trainer = Trainer(
        model=pipeline,
        args=training_args,
        train_dataset=dataset["train"],
    )
    trainer.train()