"""
Text preprocessing module for cleaning and tokenizing text data.
Located in: src/text_preprocessor.py
"""
import re
import string
import nltk
import spacy
from typing import List, Tuple, Dict
import numpy as np
from collections import Counter
from pathlib import Path

class TextPreprocessor:
    """Preprocess text data for training text generation models."""
    
    def __init__(self, sequence_length: int = 100):
        self.sequence_length = sequence_length
        self.vocab_to_int = {}
        self.int_to_vocab = {}
        self.vocab_size = 0
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("📥 Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("📥 Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
            
        # Load spaCy model (using blank model to avoid download requirements)
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ Using spaCy en_core_web_sm model")
        except OSError:
            print("⚠️ SpaCy en_core_web_sm not found, using blank model")
            self.nlp = spacy.blank("en")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        # Keep letters, numbers, spaces, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.!\?,;:"\'-]', '', text)
        
        # Fix multiple punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        # Remove excessive spaces
        text = re.sub(r' +', ' ', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def tokenize_text(self, text: str, method: str = "spacy") -> List[str]:
        """Tokenize text using specified method."""
        if method == "spacy":
            doc = self.nlp(text)
            tokens = [token.text for token in doc if not token.is_space]
        elif method == "nltk":
            tokens = nltk.word_tokenize(text)
        else:
            # Simple whitespace tokenization
            tokens = text.split()
        
        return tokens
    
    def build_vocabulary(self, texts: List[str], min_frequency: int = 2) -> Dict:
        """Build vocabulary from texts."""
        print("🔧 Building vocabulary...")
        
        # Tokenize all texts
        all_tokens = []
        for i, text in enumerate(texts):
            cleaned_text = self.clean_text(text)
            tokens = self.tokenize_text(cleaned_text)
            all_tokens.extend(tokens)
            print(f"📄 Processed text {i+1}/{len(texts)}: {len(tokens)} tokens")
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Filter by minimum frequency
        filtered_tokens = {token: count for token, count in token_counts.items() 
                         if count >= min_frequency}
        
        # Create vocabulary mappings
        # Add special tokens
        vocab_list = ['<PAD>', '<UNK>', '<START>', '<END>'] + list(filtered_tokens.keys())
        
        self.vocab_to_int = {token: i for i, token in enumerate(vocab_list)}
        self.int_to_vocab = {i: token for i, token in enumerate(vocab_list)}
        self.vocab_size = len(vocab_list)
        
        print(f"✅ Vocabulary built:")
        print(f"   - Vocabulary size: {self.vocab_size:,} tokens")
        print(f"   - Total tokens processed: {len(all_tokens):,}")
        print(f"   - Unique tokens before filtering: {len(token_counts):,}")
        print(f"   - Filtered with min_frequency={min_frequency}")
        
        return {
            'vocab_to_int': self.vocab_to_int,
            'int_to_vocab': self.int_to_vocab,
            'vocab_size': self.vocab_size,
            'token_stats': {
                'total_tokens': len(all_tokens),
                'unique_tokens': len(token_counts),
                'filtered_vocab_size': self.vocab_size
            }
        }
    
    def text_to_sequences(self, text: str) -> List[int]:
        """Convert text to sequence of integers."""
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text)
        
        # Convert tokens to integers
        sequence = []
        for token in tokens:
            if token in self.vocab_to_int:
                sequence.append(self.vocab_to_int[token])
            else:
                sequence.append(self.vocab_to_int['<UNK>'])  # Unknown token
        
        return sequence
    
    def create_training_sequences(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Create training sequences from texts."""
        print("🔢 Creating training sequences...")
        
        # Convert all texts to sequences
        all_sequences = []
        for i, text in enumerate(texts):
            sequence = self.text_to_sequences(text)
            all_sequences.extend(sequence)
            print(f"📝 Converted text {i+1}: {len(sequence)} integer tokens")
        
        print(f"📊 Total sequence length: {len(all_sequences):,} tokens")
        
        # Create input-output pairs
        X, y = [], []
        
        for i in range(0, len(all_sequences) - self.sequence_length):
            # Input sequence
            X.append(all_sequences[i:i + self.sequence_length])
            # Target (next token)
            y.append(all_sequences[i + self.sequence_length])
        
        print(f"✅ Created {len(X):,} training sequences")
        print(f"   - Input shape will be: ({len(X)}, {self.sequence_length})")
        print(f"   - Target shape will be: ({len(y)},)")
        
        return np.array(X), np.array(y)
    
    def sequences_to_text(self, sequences: List[int]) -> str:
        """Convert sequences back to text."""
        tokens = []
        for seq_id in sequences:
            if seq_id in self.int_to_vocab:
                token = self.int_to_vocab[seq_id]
                if token not in ['<PAD>', '<UNK>', '<START>', '<END>']:
                    tokens.append(token)
        
        return ' '.join(tokens)
    
    def save_preprocessing_data(self, save_dir: str):
        """Save vocabulary and preprocessing information."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        import pickle
        
        # Save vocabulary
        vocab_data = {
            'vocab_to_int': self.vocab_to_int,
            'int_to_vocab': self.int_to_vocab,
            'vocab_size': self.vocab_size,
            'sequence_length': self.sequence_length
        }
        
        with open(save_path / 'vocabulary.pkl', 'wb') as f:
            pickle.dump(vocab_data, f)
        
        print(f"💾 Preprocessing data saved to {save_path}")
    
    def load_preprocessing_data(self, save_dir: str):
        """Load vocabulary and preprocessing information."""
        save_path = Path(save_dir)
        
        import pickle
        
        with open(save_path / 'vocabulary.pkl', 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.vocab_to_int = vocab_data['vocab_to_int']
        self.int_to_vocab = vocab_data['int_to_vocab']
        self.vocab_size = vocab_data['vocab_size']
        self.sequence_length = vocab_data['sequence_length']
        
        print(f"✅ Preprocessing data loaded from {save_path}")

# Utility function for quick preprocessing
def preprocess_texts(texts: List[str], sequence_length: int = 100, min_frequency: int = 2):
    """Quick preprocessing function."""
    preprocessor = TextPreprocessor(sequence_length=sequence_length)
    
    # Build vocabulary
    vocab_info = preprocessor.build_vocabulary(texts, min_frequency=min_frequency)
    
    # Create training sequences
    X, y = preprocessor.create_training_sequences(texts)
    
    return X, y, preprocessor, vocab_info

# Example usage
if __name__ == "__main__":
    # Test with sample texts
    sample_texts = [
        "This is a sample text for testing the preprocessing pipeline.",
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing is fascinating and complex."
    ]
    
    print("🧪 Testing TextPreprocessor...")
    preprocessor = TextPreprocessor(sequence_length=10)
    vocab_info = preprocessor.build_vocabulary(sample_texts)
    X, y = preprocessor.create_training_sequences(sample_texts)
    
    print(f"✅ Training data shape: X={X.shape}, y={y.shape}")
