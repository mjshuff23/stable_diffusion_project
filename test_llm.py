# test_llm.py
from src.chat.llm_handler import QwenHandler
from src.config import Config

def test_llm():
    handler = QwenHandler()
    response = handler.generate_response("What is the capital of France?")
    print(response)

if __name__ == "__main__":
    Config.validate_paths()  # Create necessary directories
    test_llm()