import argparse
import os
from src.image_generation.train import train_stable_diffusion
from src.config import Config

def train_model(model_name: str, dataset_path: str, output_dir: str, epochs: int, batch_size: int, learning_rate: float):
    """Controller function to train different models."""
    # Update configuration dynamically
    Config.SD_MODEL_NAME = model_name
    Config.DATASET_PATH = dataset_path
    Config.OUTPUT_DIR = output_dir
    Config.TRAINING_EPOCHS = epochs
    Config.BATCH_SIZE = batch_size
    Config.LEARNING_RATE = learning_rate

    print(f"Training {model_name} on {dataset_path} for {epochs} epochs...")
    train_stable_diffusion()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for image generation.")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to fine-tune (e.g., CompVis/stable-diffusion-v1-4).")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--output", type=str, default="./trained_models", help="Directory to save trained models.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate for training.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    train_model(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
