"""
Comprehensive evaluation module for text generation models.
Located in: evaluation.py (root folder)
"""
import sys
sys.path.append('src')

import torch
import numpy as np
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

class TextGenerationEvaluator:
    """Comprehensive evaluator for text generation models."""
    
    def __init__(self, model, preprocessor, generator):
        self.model = model
        self.preprocessor = preprocessor
        self.generator = generator
        
    def calculate_perplexity(self, test_sequences: List[str]) -> float:
        """Calculate model perplexity on test data."""
        print("📊 Calculating Perplexity...")
        
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for text in test_sequences:
                # Convert to sequences
                input_ids = self.preprocessor.text_to_sequences(text)
                if len(input_ids) < 2:
                    continue
                    
                # Create input-output pairs
                for i in range(len(input_ids) - 1):
                    input_seq = input_ids[max(0, i-self.preprocessor.sequence_length+1):i+1]
                    target = input_ids[i+1]
                    
                    # Pad if necessary
                    if len(input_seq) < self.preprocessor.sequence_length:
                        pad_token = self.preprocessor.vocab_to_int.get('<PAD>', 0)
                        input_seq = [pad_token] * (self.preprocessor.sequence_length - len(input_seq)) + input_seq
                    
                    # Get model prediction
                    input_tensor = torch.tensor([input_seq])
                    if hasattr(self.model, 'init_hidden'):
                        output, _ = self.model(input_tensor)
                        logits = output[0, -1, :]
                    else:
                        output = self.model(input_tensor)
                        logits = output[0, -1, :]
                    
                    # Calculate cross-entropy loss
                    log_probs = torch.log_softmax(logits, dim=-1)
                    if target < len(log_probs):
                        loss = -log_probs[target].item()
                        total_loss += loss
                        total_tokens += 1
        
        if total_tokens == 0:
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        print(f"✅ Perplexity: {perplexity:.2f}")
        return perplexity
    
    def calculate_bleu_score(self, generated_texts: List[str], reference_texts: List[str]) -> Dict[str, float]:
        """Calculate BLEU scores (n-grams 1-4)."""
        print("📊 Calculating BLEU Scores...")
        
        def get_ngrams(text: str, n: int) -> List[tuple]:
            tokens = text.lower().split()
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        def calculate_bleu_n(generated: List[str], references: List[str], n: int) -> float:
            total_precision = 0
            total_count = 0
            
            for gen_text, ref_text in zip(generated, references):
                gen_ngrams = get_ngrams(gen_text, n)
                ref_ngrams = get_ngrams(ref_text, n)
                
                if not gen_ngrams:
                    continue
                
                ref_counter = Counter(ref_ngrams)
                matches = 0
                
                for ngram in gen_ngrams:
                    if ref_counter[ngram] > 0:
                        matches += 1
                        ref_counter[ngram] -= 1
                
                precision = matches / len(gen_ngrams)
                total_precision += precision
                total_count += 1
            
            return total_precision / max(total_count, 1)
        
        bleu_scores = {}
        for n in range(1, 5):
            score = calculate_bleu_n(generated_texts, reference_texts, n)
            bleu_scores[f'BLEU-{n}'] = score
            print(f"✅ BLEU-{n}: {score:.4f}")
        
        return bleu_scores
    
    def analyze_text_diversity(self, generated_texts: List[str]) -> Dict[str, float]:
        """Analyze lexical diversity of generated texts."""
        print("📊 Analyzing Text Diversity...")
        
        all_tokens = []
        for text in generated_texts:
            tokens = text.lower().split()
            all_tokens.extend(tokens)
        
        if not all_tokens:
            return {'diversity': 0.0, 'unique_tokens': 0, 'total_tokens': 0}
        
        unique_tokens = len(set(all_tokens))
        total_tokens = len(all_tokens)
        diversity = unique_tokens / total_tokens
        
        # Calculate repetition rate
        token_counts = Counter(all_tokens)
        repeated_tokens = sum(1 for count in token_counts.values() if count > 1)
        repetition_rate = repeated_tokens / unique_tokens
        
        diversity_metrics = {
            'lexical_diversity': diversity,
            'unique_tokens': unique_tokens,
            'total_tokens': total_tokens,
            'repetition_rate': repetition_rate,
            'vocabulary_richness': unique_tokens / max(len(generated_texts), 1)
        }
        
        print(f"✅ Lexical Diversity: {diversity:.4f}")
        print(f"✅ Repetition Rate: {repetition_rate:.4f}")
        print(f"✅ Vocabulary Richness: {diversity_metrics['vocabulary_richness']:.2f}")
        
        return diversity_metrics
    
    def benchmark_generation_methods(self, prompts: List[str]) -> Dict[str, Dict]:
        """Benchmark different generation methods."""
        print("🏁 Benchmarking Generation Methods...")
        
        methods = {
            'greedy': {'method': 'greedy'},
            'top_k_conservative': {'method': 'top_k', 'top_k': 20, 'temperature': 0.6},
            'top_k_creative': {'method': 'top_k', 'top_k': 50, 'temperature': 1.0},
            'nucleus': {'method': 'top_p', 'top_p': 0.9, 'temperature': 0.8},
        }
        
        results = {}
        
        for method_name, params in methods.items():
            print(f"\n🔄 Testing {method_name}...")
            generated_texts = []
            
            for prompt in prompts:
                try:
                    generated = self.generator.generate(
                        prompt=prompt, 
                        max_length=30,
                        **params
                    )
                    generated_texts.append(generated)
                except Exception as e:
                    print(f"Error generating with {method_name}: {e}")
                    generated_texts.append("")
            
            # Evaluate this method
            diversity = self.analyze_text_diversity(generated_texts)
            
            results[method_name] = {
                'generated_texts': generated_texts,
                'diversity': diversity,
                'params': params
            }
            
            print(f"   Diversity: {diversity['lexical_diversity']:.3f}")
        
        return results
    
    def create_evaluation_report(self, save_path: str = "evaluation_report.json"):
        """Create comprehensive evaluation report."""
        print("📋 Creating Comprehensive Evaluation Report...")
        
        # Test prompts for evaluation
        test_prompts = [
            "once upon a time in a magical kingdom",
            "the brave knight ventured into the dark forest",
            "alice found herself in a strange wonderland",
            "in the beginning there was nothing but darkness",
            "the ancient wizard opened his spellbook"
        ]
        
        # Benchmark generation methods
        method_results = self.benchmark_generation_methods(test_prompts)
        
        # Create sample reference texts for BLEU calculation
        reference_texts = [
            "once upon a time in a magical kingdom there lived a wise old king",
            "the brave knight ventured into the dark forest to find the lost treasure",
            "alice found herself in a strange wonderland full of peculiar creatures",
            "in the beginning there was nothing but darkness and then came the light",
            "the ancient wizard opened his spellbook and cast a powerful spell"
        ]
        
        # Calculate BLEU scores for nucleus method
        nucleus_texts = method_results['nucleus']['generated_texts']
        bleu_scores = self.calculate_bleu_score(nucleus_texts, reference_texts)
        
        # Sample texts for perplexity calculation
        sample_texts = [
            "alice was beginning to get very tired of sitting by her sister",
            "the rabbit hole went straight on like a tunnel for some way",
            "when the congress deems it necessary to dissolve political bands"
        ]
        
        perplexity = self.calculate_perplexity(sample_texts)
        
        # Compile comprehensive report
        report = {
            'model_info': {
                'vocab_size': self.preprocessor.vocab_size,
                'sequence_length': self.preprocessor.sequence_length,
                'model_parameters': sum(p.numel() for p in self.model.parameters())
            },
            'metrics': {
                'perplexity': perplexity,
                'bleu_scores': bleu_scores
            },
            'generation_methods': {
                method: {
                    'diversity_score': results['diversity']['lexical_diversity'],
                    'vocabulary_richness': results['diversity']['vocabulary_richness'],
                    'sample_text': results['generated_texts'][0] if results['generated_texts'] else ""
                }
                for method, results in method_results.items()
            }
        }
        
        # Save report
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Evaluation report saved to: {save_path}")
        
        # Print summary
        self._print_report_summary(report)
        
        return report
    
    def _print_report_summary(self, report):
        """Print a formatted summary of the evaluation report."""
        print("\n" + "="*60)
        print("📊 EVALUATION REPORT SUMMARY")
        print("="*60)
        
        print(f"\n🧠 Model Information:")
        print(f"   Vocabulary Size: {report['model_info']['vocab_size']:,}")
        print(f"   Parameters: {report['model_info']['model_parameters']:,}")
        
        print(f"\n📈 Key Metrics:")
        print(f"   Perplexity: {report['metrics']['perplexity']:.2f}")
        for bleu_type, score in report['metrics']['bleu_scores'].items():
            print(f"   {bleu_type}: {score:.4f}")
        
        print(f"\n🎯 Generation Methods Ranking:")
        methods = sorted(report['generation_methods'].items(), 
                        key=lambda x: x[1]['diversity_score'], reverse=True)
        
        for i, (method, metrics) in enumerate(methods, 1):
            print(f"   {i}. {method}: Diversity={metrics['diversity_score']:.3f}")
        
        print("\n" + "="*60)

def main():
    """Run comprehensive evaluation."""
    print("🚀 COMPREHENSIVE MODEL EVALUATION")
    print("="*50)
    
    # Load the trained model and preprocessor
    from data_pipeline import DataPipeline
    from model import create_model, MODEL_CONFIGS
    from trainer import ModelTrainer, load_model
    from text_preprocessor import TextPreprocessor
    from generator import TextGenerator
    
    # Load preprocessor
    print("📥 Loading preprocessor...")
    preprocessor = TextPreprocessor(sequence_length=50)
    preprocessor.load_preprocessing_data("data/preprocessed")
    
    # Load trained model
    print("📥 Loading trained model...")
    checkpoint = load_model("models/lstm_text_generator_best.pth")
    model = create_model('lstm', checkpoint['vocab_size'], **MODEL_CONFIGS['lstm_medium'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create generator
    print("🤖 Creating text generator...")
    generator = TextGenerator(model, preprocessor)
    
    # Create evaluator
    evaluator = TextGenerationEvaluator(model, preprocessor, generator)
    
    # Run comprehensive evaluation
    report = evaluator.create_evaluation_report("evaluation_report.json")
    
    print("\n🎉 Comprehensive evaluation completed!")
    print("📄 Report saved to: evaluation_report.json")

if __name__ == "__main__":
    main()
