from fastapi import FastAPI
from src.image_generation.stable_diffusion import ImageGenerator
from src.config import Config
from src.chat.llm_handler import QwenHandler
QwenHandler

app = FastAPI()
image_generator = ImageGenerator()
chat_handler = QwenHandler()

@app.post("/generate-image")
async def generate_image(prompt: str):
    image = image_generator.generate_image(prompt)
    return {"status": "success", "image": image}

@app.post("/chat")
async def chat(message: str):
    response = chat_handler.generate_response(message)
    return {"status": "success", "response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)