"""
Interactive demo for text generation model.
Located in: interactive_demo.py (root folder)
"""
import sys
sys.path.append('src')

import torch
import json
from pathlib import Path

class InteractiveTextGenerator:
    """Interactive interface for text generation."""
    
    def __init__(self):
        self.model = None
        self.generator = None
        self.preprocessor = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and components."""
        try:
            print("🤖 Loading trained model...")
            
            # Load components
            from model import create_model, MODEL_CONFIGS
            from trainer import load_model
            from text_preprocessor import TextPreprocessor
            from generator import TextGenerator
            
            # Load preprocessor
            self.preprocessor = TextPreprocessor(sequence_length=50)
            self.preprocessor.load_preprocessing_data("data/preprocessed")
            
            # Load trained model
            checkpoint = load_model("models/lstm_text_generator_best.pth")
            self.model = create_model('lstm', checkpoint['vocab_size'], **MODEL_CONFIGS['lstm_medium'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Create generator
            self.generator = TextGenerator(self.model, self.preprocessor, device='cpu')
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please ensure you have trained the model first by running: python train_model.py")
            self.model = None
    
    def show_menu(self):
        """Display the main menu."""
        print("\n" + "="*60)
        print("🎮 INTERACTIVE TEXT GENERATION DEMO")
        print("="*60)
        print("Choose an option:")
        print("1. 📝 Generate text with custom prompt")
        print("2. 🎲 Generate random story")  
        print("3. 🎯 Compare generation methods")
        print("4. 📚 Try preset story prompts")
        print("5. 📊 View model information")
        print("6. ❌ Exit")
        print("="*60)
    
    def custom_generation(self):
        """Generate text with user's custom prompt."""
        if not self.model:
            print("❌ Model not loaded. Please train the model first.")
            return
        
        print("\n📝 Custom Text Generation")
        print("-" * 30)
        
        prompt = input("Enter your prompt: ").strip()
        if not prompt:
            prompt = "Once upon a time"
        
        print("\nChoose generation method:")
        print("1. Greedy (deterministic)")
        print("2. Top-k sampling (balanced)")
        print("3. Nucleus sampling (creative)")
        
        choice = input("Choice (1-3): ").strip()
        
        # Set parameters based on choice
        if choice == '1':
            params = {'method': 'greedy'}
        elif choice == '2':
            params = {'method': 'top_k', 'top_k': 30, 'temperature': 0.8}
        elif choice == '3':
            params = {'method': 'top_p', 'top_p': 0.9, 'temperature': 0.8}
        else:
            params = {'method': 'top_k', 'top_k': 30, 'temperature': 0.8}
        
        max_length = int(input("Max length (10-100): ") or "50")
        
        print("\n🤖 Generating...")
        try:
            generated = self.generator.generate(
                prompt=prompt,
                max_length=max_length,
                **params
            )
            
            print(f"\n✨ Generated Text:")
            print(f"'{generated}'")
            
        except Exception as e:
            print(f"❌ Error generating text: {e}")
    
    def random_story(self):
        """Generate a random story."""
        if not self.model:
            print("❌ Model not loaded.")
            return
        
        import random
        
        story_starters = [
            "Once upon a time in a magical forest",
            "The brave adventurer discovered a hidden cave",
            "In the ancient library, Alice found a mysterious book",
            "The wise old wizard looked into his crystal ball",
            "Deep in the enchanted garden, something stirred"
        ]
        
        prompt = random.choice(story_starters)
        print(f"\n🎲 Random Story Generator")
        print(f"Prompt: '{prompt}'")
        print("\n🤖 Generating story...")
        
        try:
            generated = self.generator.generate(
                prompt=prompt,
                max_length=60,
                method='top_k',
                top_k=40,
                temperature=0.9
            )
            
            print(f"\n📚 Your Random Story:")
            print(f"'{generated}'")
            
        except Exception as e:
            print(f"❌ Error generating story: {e}")
    
    def compare_methods(self):
        """Compare different generation methods."""
        if not self.model:
            print("❌ Model not loaded.")
            return
        
        prompt = input("\nEnter prompt to compare (or press Enter for default): ").strip()
        if not prompt:
            prompt = "Alice wandered through the mysterious garden"
        
        print(f"\n🎯 Comparing generation methods for: '{prompt}'")
        print("=" * 50)
        
        methods = {
            'Greedy': {'method': 'greedy'},
            'Conservative': {'method': 'top_k', 'top_k': 20, 'temperature': 0.6},
            'Balanced': {'method': 'top_k', 'top_k': 40, 'temperature': 0.8},
            'Creative': {'method': 'top_p', 'top_p': 0.9, 'temperature': 1.0}
        }
        
        for name, params in methods.items():
            print(f"\n{name} Method:")
            try:
                generated = self.generator.generate(
                    prompt=prompt,
                    max_length=40,
                    **params
                )
                print(f"  '{generated}'")
            except Exception as e:
                print(f"  Error: {e}")
    
    def preset_prompts(self):
        """Try preset story prompts."""
        if not self.model:
            print("❌ Model not loaded.")
            return
        
        prompts = {
            "1": "Alice fell down the rabbit hole and found",
            "2": "The Declaration of Independence states that",
            "3": "In the beginning, there was only darkness until",
            "4": "The magical spell book contained secrets about",
            "5": "The brave knight faced the dragon and"
        }
        
        print("\n📚 Preset Story Prompts")
        print("-" * 25)
        for key, prompt in prompts.items():
            print(f"{key}. {prompt}")
        
        choice = input("\nChoose a prompt (1-5): ").strip()
        
        if choice in prompts:
            prompt = prompts[choice]
            print(f"\n🤖 Generating for: '{prompt}'")
            
            try:
                generated = self.generator.generate(
                    prompt=prompt,
                    max_length=60,
                    method='top_k',
                    top_k=35,
                    temperature=0.8
                )
                
                print(f"\n✨ Generated:")
                print(f"'{generated}'")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("Invalid choice!")
    
    def show_model_info(self):
        """Display model information."""
        if not self.model:
            print("❌ Model not loaded.")
            return
        
        print("\n📊 Model Information")
        print("-" * 20)
        print(f"Model Type: LSTM Text Generator")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Vocabulary Size: {self.preprocessor.vocab_size:,}")
        print(f"Sequence Length: {self.preprocessor.sequence_length}")
        
        # Load evaluation report if available
        try:
            with open('evaluation_report.json', 'r') as f:
                report = json.load(f)
            print(f"\n📈 Performance Metrics:")
            print(f"Perplexity: {report['metrics']['perplexity']:.2f}")
            for bleu_type, score in report['metrics']['bleu_scores'].items():
                print(f"{bleu_type}: {score:.4f}")
        except FileNotFoundError:
            print("\n📈 Run 'python evaluation.py' for detailed performance metrics")
    
    def run(self):
        """Main interactive loop."""
        if not self.model:
            return
        
        while True:
            try:
                self.show_menu()
                choice = input("\nEnter your choice (1-6): ").strip()
                
                if choice == '1':
                    self.custom_generation()
                elif choice == '2':
                    self.random_story()
                elif choice == '3':
                    self.compare_methods()
                elif choice == '4':
                    self.preset_prompts()
                elif choice == '5':
                    self.show_model_info()
                elif choice == '6':
                    print("\n👋 Thank you for using the Text Generator!")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Run the interactive demo."""
    demo = InteractiveTextGenerator()
    demo.run()

if __name__ == "__main__":
    main()
