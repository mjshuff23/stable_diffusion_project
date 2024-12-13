# type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, List, Optional
from src.config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenHandler:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Qwen LLM handler.
        Args:
            model_path: Optional path to a fine-tuned model. If None, uses the default from Config.
        """
        self.model_path = model_path or Config.LLM_MODEL_NAME
        self.device = torch.device(Config.DEVICE)
        
        logger.info(f"Loading Qwen model from {self.model_path}")
        
        # Load tokenizer and model with Qwen-specific settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=Config.TRUST_REMOTE_CODE,
            cache_dir=Config.CACHE_DIR
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",  # Automatically handle model parallelism
            trust_remote_code=Config.TRUST_REMOTE_CODE,
            torch_dtype=torch.float16,  # Use half precision for memory efficiency
            cache_dir=Config.CACHE_DIR
        )
        
        logger.info("Model loaded successfully")
    
    def generate_response(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response using the Qwen model.
        
        Args:
            prompt: The user's input prompt
            max_length: Optional override for response length
            temperature: Optional override for temperature
            top_p: Optional override for top_p sampling
            system_prompt: Optional system prompt to guide the model's behavior
        
        Returns:
            str: The generated response
        """
        # Prepare the full prompt with optional system prompt
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        # Tokenize the input
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        # Generate response with specified or default parameters
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length or Config.MAX_LENGTH,
                temperature=temperature or Config.TEMPERATURE,
                top_p=top_p or Config.TOP_P,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode and return the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from the response
        response = response[len(full_prompt):].strip()
        
        return response
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Handle a chat conversation with message history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            system_prompt: Optional system prompt to guide the conversation
        
        Returns:
            str: The model's response
        """
        # Format the conversation history
        conversation = []
        if system_prompt:
            conversation.append(f"System: {system_prompt}\n")
            
        for message in messages:
            role = message['role']
            content = message['content']
            if role == 'user':
                conversation.append(f"User: {content}")
            elif role == 'assistant':
                conversation.append(f"Assistant: {content}")
                
        # Join the conversation history
        full_prompt = "\n".join(conversation)
        
        # Generate and return the response
        return self.generate_response(full_prompt)
    
    def stream_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ):
        """
        Stream the response token by token.
        
        Args:
            prompt: The user's input prompt
            system_prompt: Optional system prompt
            
        Yields:
            str: Generated tokens one at a time
        """
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        # Configure streamed generation
        stream = self.model.generate(
            inputs.input_ids,
            max_length=Config.MAX_LENGTH,
            temperature=Config.TEMPERATURE,
            top_p=Config.TOP_P,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            streamer=True
        )
        
        for token in stream:
            yield self.tokenizer.decode([token], skip_special_tokens=True)
