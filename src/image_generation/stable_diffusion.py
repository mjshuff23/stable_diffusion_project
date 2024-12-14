from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from ..config import Config

class ImageGenerator:
    def __init__(self):
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            Config.SD_MODEL_NAME,
            torch_dtype=torch.float16
        ).to("cuda")
    
    def generate_image(self, prompt: str) -> Image.Image:
        """Generate image from text prompt."""
        with torch.no_grad():
            image = self.pipeline(prompt).images[0]
        return image