"""
Neural network models for text generation.
Located in: src/model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class LSTMTextGenerator(nn.Module):
    """LSTM-based text generation model."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, hidden_dim: int = 512, 
                 num_layers: int = 2, dropout: float = 0.3):
        super(LSTMTextGenerator, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        """Forward pass through the LSTM model."""
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)  # (batch_size, seq_len, hidden_dim)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Output layer - get predictions for all time steps
        output = self.fc(lstm_out)  # (batch_size, seq_len, vocab_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden and cell states."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

class TransformerTextGenerator(nn.Module):
    """Transformer-based text generation model."""
    
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, 
                 num_layers: int = 6, dim_feedforward: int = 2048, 
                 max_seq_length: int = 512, dropout: float = 0.1):
        super(TransformerTextGenerator, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output layer
        self.fc = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        """Forward pass through the Transformer model."""
        seq_len = x.size(1)
        
        # Create causal mask if not provided
        if mask is None:
            mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Embedding and positional encoding
        embedded = self.embedding(x) * math.sqrt(self.d_model)
        embedded = self.pos_encoding(embedded)
        
        # Transformer decoder (using embedded as both tgt and memory for autoregressive)
        output = self.transformer_decoder(embedded, embedded, tgt_mask=mask)
        
        # Output layer
        output = self.fc(output)  # (batch_size, seq_len, vocab_size)
        
        return output
    
    def _generate_square_subsequent_mask(self, sz: int):
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class HybridTextGenerator(nn.Module):
    """Hybrid model combining LSTM and Transformer components."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, 
                 lstm_hidden: int = 512, transformer_dim: int = 512,
                 num_transformer_layers: int = 4, dropout: float = 0.2):
        super(HybridTextGenerator, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Shared embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM component
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # Projection layer from LSTM to Transformer
        self.lstm_to_transformer = nn.Linear(lstm_hidden, transformer_dim)
        
        # Transformer component
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=8,
            dim_feedforward=transformer_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_transformer_layers)
        
        # Output layer
        self.fc = nn.Linear(transformer_dim, vocab_size)
        
    def forward(self, x, lstm_hidden=None):
        """Forward pass through the hybrid model."""
        batch_size = x.size(0)
        
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM processing
        lstm_out, lstm_hidden = self.lstm(embedded, lstm_hidden)
        
        # Project LSTM output to transformer dimension
        transformer_input = self.lstm_to_transformer(lstm_out)
        
        # Transformer processing
        transformer_out = self.transformer(transformer_input)
        
        # Output layer
        output = self.fc(transformer_out)
        
        return output, lstm_hidden

def create_model(model_type: str, vocab_size: int, **kwargs):
    """Factory function to create different model types."""
    
    if model_type.lower() == 'lstm':
        return LSTMTextGenerator(
            vocab_size=vocab_size,
            embedding_dim=kwargs.get('embedding_dim', 256),
            hidden_dim=kwargs.get('hidden_dim', 512),
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.3)
        )
    
    elif model_type.lower() == 'transformer':
        return TransformerTextGenerator(
            vocab_size=vocab_size,
            d_model=kwargs.get('d_model', 512),
            nhead=kwargs.get('nhead', 8),
            num_layers=kwargs.get('num_layers', 6),
            dim_feedforward=kwargs.get('dim_feedforward', 2048),
            max_seq_length=kwargs.get('max_seq_length', 512),
            dropout=kwargs.get('dropout', 0.1)
        )
    
    elif model_type.lower() == 'hybrid':
        return HybridTextGenerator(
            vocab_size=vocab_size,
            embedding_dim=kwargs.get('embedding_dim', 256),
            lstm_hidden=kwargs.get('lstm_hidden', 512),
            transformer_dim=kwargs.get('transformer_dim', 512),
            num_transformer_layers=kwargs.get('num_transformer_layers', 4),
            dropout=kwargs.get('dropout', 0.2)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Model configuration presets
MODEL_CONFIGS = {
    'lstm_small': {
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.3
    },
    'lstm_medium': {
        'embedding_dim': 256,
        'hidden_dim': 512,
        'num_layers': 2,
        'dropout': 0.3
    },
    'lstm_large': {
        'embedding_dim': 512,
        'hidden_dim': 1024,
        'num_layers': 3,
        'dropout': 0.4
    },
    'transformer_small': {
        'd_model': 256,
        'nhead': 4,
        'num_layers': 4,
        'dim_feedforward': 1024,
        'dropout': 0.1
    },
    'transformer_medium': {
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1
    }
}

# Example usage
if __name__ == "__main__":
    # Test model creation
    vocab_size = 1000
    
    print("🧪 Testing model creation...")
    
    # Test LSTM model
    lstm_model = create_model('lstm', vocab_size, **MODEL_CONFIGS['lstm_medium'])
    print(f"✅ LSTM Model created: {sum(p.numel() for p in lstm_model.parameters()):,} parameters")
    
    # Test Transformer model
    transformer_model = create_model('transformer', vocab_size, **MODEL_CONFIGS['transformer_small'])
    print(f"✅ Transformer Model created: {sum(p.numel() for p in transformer_model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size, seq_len = 32, 50
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        lstm_output, _ = lstm_model(dummy_input)
        transformer_output = transformer_model(dummy_input)
        
    print(f"✅ LSTM output shape: {lstm_output.shape}")
    print(f"✅ Transformer output shape: {transformer_output.shape}")
    print("🎉 All models working correctly!")
