"""
Text generation module for inference and creative text generation.
Located in: src/generator.py
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Union
import random
import sys
from pathlib import Path

class TextGenerator:
    """Advanced text generator with multiple sampling strategies."""
    
    def __init__(self, model, preprocessor, device: str = 'auto'):
        self.model = model
        self.preprocessor = preprocessor
        
        # Auto-detect device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        self.model.to(self.device)
        self.model.eval()
        
        print(f"🤖 TextGenerator initialized on {self.device}")
    
    def generate(self, prompt: str = "", max_length: int = 100, 
                method: str = "top_k", temperature: float = 0.8,
                top_k: int = 50, top_p: float = 0.9, 
                repetition_penalty: float = 1.1,
                stop_tokens: Optional[List[str]] = None) -> str:
        """Generate text using various sampling strategies."""
        
        print(f"🎯 Generating text with method: {method}")
        print(f"📝 Prompt: '{prompt}'")
        
        # Handle empty prompt
        if not prompt.strip():
            prompt = self._get_random_seed()
            print(f"🎲 Using random seed: '{prompt}'")
        
        # Convert prompt to token sequence
        input_ids = self.preprocessor.text_to_sequences(prompt)
        
        # Ensure we have a minimum sequence length
        sequence_length = self.preprocessor.sequence_length
        if len(input_ids) < sequence_length:
            padding_needed = sequence_length - len(input_ids)
            pad_token = self.preprocessor.vocab_to_int.get('<PAD>', 0)
            input_ids = [pad_token] * padding_needed + input_ids
        
        # Take only the last sequence_length tokens
        input_ids = input_ids[-sequence_length:]
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        generated_tokens = []
        past_tokens = input_ids.copy()
        
        # Generation loop
        with torch.no_grad():
            for step in range(max_length):
                # Get model predictions
                if hasattr(self.model, 'init_hidden'):
                    output, _ = self.model(input_tensor)
                    logits = output[0, -1, :]
                else:
                    output = self.model(input_tensor)
                    logits = output[0, -1, :]
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    logits = self._apply_repetition_penalty(
                        logits, past_tokens, repetition_penalty
                    )
                
                # Sample next token
                if method == 'greedy':
                    next_token = torch.argmax(logits).item()
                elif method == 'top_k':
                    next_token = self._top_k_sampling(logits, top_k, temperature)
                elif method == 'top_p':
                    next_token = self._nucleus_sampling(logits, top_p, temperature)
                elif method == 'temperature':
                    next_token = self._temperature_sampling(logits, temperature)
                else:
                    raise ValueError(f"Unknown sampling method: {method}")
                
                # Convert token back to text
                if next_token in self.preprocessor.int_to_vocab:
                    token_text = self.preprocessor.int_to_vocab[next_token]
                    
                    if stop_tokens and token_text in stop_tokens:
                        break
                    
                    if token_text not in ['<PAD>', '<UNK>', '<START>', '<END>']:
                        generated_tokens.append(token_text)
                    
                    past_tokens.append(next_token)
                    
                    # Update input for next iteration
                    new_input = torch.cat([
                        input_tensor[:, 1:], 
                        torch.tensor([[next_token]], device=self.device)
                    ], dim=1)
                    input_tensor = new_input
                else:
                    break
        
        generated_text = ' '.join(generated_tokens)
        final_text = self._post_process_text(prompt + ' ' + generated_text)
        
        print(f"✅ Generated {len(generated_tokens)} tokens")
        return final_text
    
    def _get_random_seed(self) -> str:
        seed_phrases = [
            "Once upon a time", "In a distant land", "The wizard cast a spell",
            "Deep in the forest", "Long ago", "There was a kingdom"
        ]
        return random.choice(seed_phrases)
    
    def _apply_repetition_penalty(self, logits, past_tokens, penalty):
        unique_tokens = set(past_tokens)
        for token in unique_tokens:
            if token < len(logits):
                if logits[token] > 0:
                    logits[token] /= penalty
                else:
                    logits[token] *= penalty
        return logits
    
    def _temperature_sampling(self, logits, temperature):
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, 1).item()
    
    def _top_k_sampling(self, logits, k, temperature):
        top_k_logits, top_k_indices = torch.topk(logits, k)
        scaled_logits = top_k_logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        selected_index = torch.multinomial(probs, 1).item()
        return top_k_indices[selected_index].item()
    
    def _nucleus_sampling(self, logits: torch.Tensor, p: float, temperature: float) -> int:
        """Sample using nucleus (top-p) sampling."""
        scaled_logits = logits / temperature
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        cutoff_index = torch.where(cumulative_probs > p)[0]
        if len(cutoff_index) > 0:
            cutoff_index = cutoff_index[0].item()
        else:
            cutoff_index = len(sorted_logits)
            
        #Fix: Ensure we have at leat 1 toke to sample from 
        cutoff_index = max(1, cutoff_index)
        
        top_p_logits = sorted_logits[:cutoff_index]
        top_p_indices = sorted_indices[:cutoff_index]
        
        probs = F.softmax(top_p_logits, dim=-1)
        selected_index = torch.multinomial(probs, 1).item()
        return top_p_indices[selected_index].item()
    
    def _post_process_text(self, text):
        text = text.strip()
        text = text.replace(' ,', ',').replace(' .', '.').replace(' !', '!')
        text = text.replace(' ?', '?').replace(' ;', ';').replace(' :', ':')
        
        sentences = text.split('. ')
        sentences = [s.capitalize() for s in sentences]
        text = '. '.join(sentences)
        return text
    
    def interactive_generation(self):
        """Interactive text generation session."""
        print("🎮 INTERACTIVE TEXT GENERATION")
        print("Type 'quit' to exit")
        
        while True:
            try:
                user_input = input("\nEnter prompt: ").strip()
                if user_input.lower() in ['quit', 'exit']:
                    break
                    
                generated = self.generate(prompt=user_input, max_length=50)
                print(f"\n🤖 Generated: '{generated}'")
                
            except KeyboardInterrupt:
                break

def load_generator_from_checkpoint(model_path: str, preprocessor_path: str):
    """Load a text generator from saved model and preprocessor."""
    # Load preprocessor
    import pickle
    with open(preprocessor_path, 'rb') as f:
        preprocessor_data = pickle.load(f)
    
    # Create preprocessor instance
    from text_preprocessor import TextPreprocessor
    preprocessor = TextPreprocessor()
    preprocessor.vocab_to_int = preprocessor_data['vocab_to_int']
    preprocessor.int_to_vocab = preprocessor_data['int_to_vocab']
    preprocessor.vocab_size = preprocessor_data['vocab_size']
    preprocessor.sequence_length = preprocessor_data['sequence_length']
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model
    from model import create_model
    model = create_model(
        model_type='lstm',
        vocab_size=checkpoint['vocab_size']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return TextGenerator(model, preprocessor)

# Example usage
if __name__ == "__main__":
    print("🧪 Testing TextGenerator...")
    print("✅ TextGenerator module ready!")
    print("📝 Note: Requires trained model and preprocessor to function")
