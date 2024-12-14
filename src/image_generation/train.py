from diffusers import StableDiffusionPipeline
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from datasets import load_dataset
from ..config import Config

def train_stable_diffusion():
    """Fine-tune Stable Diffusion on a custom dataset."""
    # Define data transformations
    transform = Compose([
        Resize((512, 512)),
        CenterCrop(512),
        ToTensor(),
        Normalize([0.5], [0.5])
    ]) 

    def preprocess_data(example):
        """Apply transformations to each image in the dataset."""
        if isinstance(example["image"], list):  # Handle batched images
            example["pixel_values"] = [transform(img.convert("RGB")) for img in example["image"]]
        else:  # Handle single image
            example["pixel_values"] = transform(example["image"].convert("RGB"))
        return example

    # Load dataset
    dataset = load_dataset("imagefolder", data_dir=Config.DATASET_PATH)
    dataset = dataset.with_transform(preprocess_data)

    # DataLoader
    train_loader = DataLoader(
        dataset["train"],
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: torch.stack([b["pixel_values"] for b in batch])
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pipeline and components
    pipeline = StableDiffusionPipeline.from_pretrained(
        Config.SD_MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)

    vae = pipeline.vae
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    unet = pipeline.unet

    optimizer = torch.optim.AdamW(unet.parameters(), lr=Config.LEARNING_RATE)

    for epoch in range(Config.TRAINING_EPOCHS):
        unet.train()
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Move images to device and encode to latents
            pixel_values = batch.to(device)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215  # Latent scaling

            # Generate random noise and add to latents
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()
            noisy_latents = latents + noise

            # Generate text embeddings for conditioning
            text_inputs = tokenizer(
                ["A placeholder prompt"] * latents.shape[0],
                return_tensors="pt", padding=True
            ).input_ids.to(device)
            with torch.no_grad():
                text_embeddings = text_encoder(text_inputs).last_hidden_state

            # Predict the noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample

            # Compute the loss (MSE loss between predicted noise and actual noise)
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    pipeline.save_pretrained(Config.MODEL_SAVE_DIR)
