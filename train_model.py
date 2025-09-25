"""
Complete training script that ties everything together.
Located in: train_model.py (root folder)
"""
import sys
sys.path.append('src')

import numpy as np
import torch
from data_pipeline import DataPipeline
from model import create_model, MODEL_CONFIGS
from trainer import ModelTrainer
from generator import TextGenerator

def main():
    print("🚀 COMPLETE TEXT GENERATION MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Load preprocessed data
    print("\n📥 Step 1: Loading Data")
    pipeline = DataPipeline(data_dir="data", sequence_length=50)
    X, y, summary = pipeline.load_processed_data()
    
    if X is None:
        print("❌ No processed data found. Run test_pipeline.py first!")
        return
    
    # Step 2: Create model
    print("\n🧠 Step 2: Creating Model")
    vocab_size = summary['vocab_size']
    
    # Choose model type (you can change this)
    model_type = 'lstm'  # or 'transformer'
    config = MODEL_CONFIGS['lstm_medium']
    
    model = create_model(model_type, vocab_size, **config)
    print(f"✅ Created {model_type} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Step 3: Train model
    print("\n🎯 Step 3: Training Model")
    trainer = ModelTrainer(model, vocab_size)
    
    training_results = trainer.train(
        X_train=X,
        y_train=y,
        epochs=20,
        batch_size=32,
        learning_rate=0.001,
        validation_split=0.1,
        save_dir="models",
        model_name=f"{model_type}_text_generator",
        patience=5
    )
    
    print(f"\n🏆 Training completed!")
    print(f"   Best validation loss: {training_results['best_val_loss']:.4f}")
    
    # Step 4: Test text generation
    print("\n🤖 Step 4: Testing Text Generation")
    
    # Load preprocessor
    from text_preprocessor import TextPreprocessor
    preprocessor = TextPreprocessor(sequence_length=50)
    preprocessor.load_preprocessing_data("data/preprocessed")
    
    # Create generator
    generator = TextGenerator(model, preprocessor)
    
    # Generate sample texts
    test_prompts = [
        "once upon a time",
        "in the beginning",
        "the Declaration of Independence"
    ]
    
    for prompt in test_prompts:
        generated = generator.generate(
            prompt=prompt,
            max_length=50,
            method='top_k',
            temperature=0.8
        )
        print(f"\n📝 Prompt: '{prompt}'")
        print(f"🤖 Generated: '{generated}'")
    
    print("\n🎉 Training and testing completed successfully!")
    print(f"💾 Model saved to: models/{model_type}_text_generator_best.pth")

if __name__ == "__main__":
    main()
