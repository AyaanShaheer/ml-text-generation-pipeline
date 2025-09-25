"""
Complete data pipeline for text generation model.
Combines data collection and preprocessing.
Located in: src/data_pipeline.py
"""
import sys
import os
from pathlib import Path
import numpy as np
import pickle
from typing import List, Tuple, Dict

# Add current directory to path for relative imports
current_dir = Path(__file__).parent if '__file__' in globals() else Path('src')
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

class DataPipeline:
    """Complete pipeline for preparing text data."""
    
    def __init__(self, data_dir: str = "data", sequence_length: int = 100):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        
        # Initialize components
        self.collector = None
        self.preprocessor = None
        
        # Create directories
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "processed").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "preprocessed").mkdir(parents=True, exist_ok=True)
    
    def run_full_pipeline(self, num_books: int = 3, min_frequency: int = 2) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Run the complete data pipeline."""
        print("=" * 60)
        print("🚀 STARTING COMPLETE DATA PIPELINE")
        print("=" * 60)
        
        # Step 1: Data Collection
        print("\n📚 Step 1: Data Collection")
        print("-" * 40)
        
        try:
            # Import here to avoid circular imports
            from data_collector import GutenbergDataCollector
            self.collector = GutenbergDataCollector(str(self.data_dir))
            
            # Check if we have existing texts
            processed_files = list((self.data_dir / "processed").glob("book_*.txt"))
            
            if len(processed_files) >= num_books:
                print(f"✅ Found {len(processed_files)} existing books, loading them...")
                texts = []
                for file_path in processed_files[:num_books]:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        texts.append(content)
                        print(f"📖 Loaded: {file_path.name} ({len(content):,} chars)")
            else:
                print(f"📥 Collecting {num_books} books from Project Gutenberg...")
                texts = self.collector.collect_books(num_books=num_books)
                
                if not texts:
                    raise Exception("No texts collected from Project Gutenberg")
                    
        except Exception as e:
            print(f"⚠️ Error in data collection: {e}")
            print("🔄 Using fallback sample data...")
            texts = self._get_sample_data()
        
        # Step 2: Text Preprocessing
        print("\n🔧 Step 2: Text Preprocessing")
        print("-" * 40)
        
        try:
            from text_preprocessor import TextPreprocessor
            self.preprocessor = TextPreprocessor(sequence_length=self.sequence_length)
            
            # Build vocabulary
            vocab_info = self.preprocessor.build_vocabulary(texts, min_frequency=min_frequency)
            
            # Create training sequences
            X, y = self.preprocessor.create_training_sequences(texts)
            
            # Save preprocessing data
            self.preprocessor.save_preprocessing_data(str(self.data_dir / "preprocessed"))
            
        except Exception as e:
            print(f"❌ Error in preprocessing: {e}")
            raise
        
        # Step 3: Data Summary and Validation
        print("\n📊 Step 3: Data Summary & Validation")
        print("-" * 40)
        
        summary = {
            'num_books': len(texts),
            'total_characters': sum(len(text) for text in texts),
            'vocab_size': vocab_info['vocab_size'],
            'sequence_length': self.sequence_length,
            'num_sequences': len(X),
            'X_shape': X.shape,
            'y_shape': y.shape,
            'sample_vocab': list(self.preprocessor.vocab_to_int.keys())[:20],
            'min_frequency': min_frequency
        }
        
        self._print_summary(summary)
        
        # Save summary
        with open(self.data_dir / "data_summary.pkl", 'wb') as f:
            pickle.dump(summary, f)
        
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return X, y, summary
    
    def _get_sample_data(self) -> List[str]:
        """Fallback sample data for testing."""
        print("📝 Using built-in sample literary texts...")
        return [
            """
            Once upon a time, in a land far away, there lived a wise old wizard who possessed great magical powers. 
            He spent his days studying ancient books and brewing mysterious potions in his tower. The wizard was known 
            throughout the kingdom for his kindness and wisdom. People would travel from distant villages to seek his 
            advice on matters both magical and mundane. His tower, perched high on a hill, was filled with scrolls, 
            crystal balls, and bubbling cauldrons. The wizard had a pet raven named Cornelius who could speak in riddles 
            and often helped visitors solve their problems. Together, they maintained the balance of magic in the realm 
            and protected the innocent from dark forces that sometimes threatened the peace. The wizard's library contained 
            thousands of ancient tomes, each filled with spells, potions, and wisdom from ages past. Students would come 
            from far and wide to learn from the master, and he taught them not only magic, but also compassion and justice.
            """,
            
            """
            The bustling marketplace was filled with merchants selling their wares from colorful stalls. The aroma of 
            freshly baked bread mixed with exotic spices created an intoxicating atmosphere. Children ran between the 
            stalls while their parents haggled with vendors over prices. Musicians played lively tunes on street corners, 
            their hat filled with coins from generous passersby. The cobblestone streets echoed with the sounds of horses' 
            hooves and cart wheels. As the sun began to set, lanterns were lit, casting a warm glow over the entire market. 
            This was the heart of the city, where people from all walks of life came together to trade, socialize, and 
            share stories from their travels. The marketplace had been the center of commerce for centuries, with generations 
            of families running the same stalls, passing down their trade secrets and building lasting relationships with 
            customers who became more like family than mere buyers and sellers.
            """,
            
            """
            In the depths of the ancient forest, where sunlight barely penetrated the thick canopy of leaves, magical 
            creatures made their home. Fairies danced among the flowers, their wings shimmering like dewdrops in the 
            morning light. Ancient trees whispered secrets to those who knew how to listen, their branches swaying gently 
            in the breeze. A crystal-clear stream wound its way through the forest, providing fresh water to all the 
            woodland inhabitants. The forest was protected by an old magic that kept harmful influences at bay, making it 
            a sanctuary for creatures both large and small. Those who entered with pure hearts often found exactly what 
            they were seeking, whether it was wisdom, healing, or simply a moment of peace. The forest had stood for 
            millennia, witnessing the rise and fall of kingdoms, yet remaining unchanged and eternal. Druids and nature 
            spirits worked together to maintain the delicate balance, ensuring that the forest would continue to thrive 
            for generations to come, a testament to the power of harmony between magic and nature.
            """
        ]
    
    def _print_summary(self, summary: Dict):
        """Print a formatted summary of the data pipeline results."""
        print("📋 DATASET SUMMARY:")
        print(f"   📖 Number of books: {summary['num_books']}")
        print(f"   📄 Total characters: {summary['total_characters']:,}")
        print(f"   🔤 Vocabulary size: {summary['vocab_size']:,}")
        print(f"   📏 Sequence length: {summary['sequence_length']}")
        print(f"   🔢 Training sequences: {summary['num_sequences']:,}")
        print(f"   📊 Data shapes: X={summary['X_shape']}, y={summary['y_shape']}")
        
        # Calculate approximate dataset size
        dataset_size_mb = (summary['X_shape'][0] * summary['X_shape'][1] * 4) / (1024 * 1024)  # int32
        print(f"   💾 Dataset size: ~{dataset_size_mb:.2f} MB")
        
        print(f"   🎯 Sample vocab: {', '.join(summary['sample_vocab'][:8])}...")
        
        # Validation checks
        print("\n✅ VALIDATION CHECKS:")
        if summary['vocab_size'] > 100:
            print("   ✅ Vocabulary size is adequate for training")
        else:
            print("   ⚠️ Vocabulary size might be too small")
            
        if summary['num_sequences'] > 1000:
            print("   ✅ Sufficient training sequences")
        else:
            print("   ⚠️ Limited training sequences - consider more data")
    
    def load_processed_data(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load previously processed data."""
        try:
            # Load training data
            X = np.load(self.data_dir / "preprocessed" / "X_train.npy")
            y = np.load(self.data_dir / "preprocessed" / "y_train.npy")
            
            # Load summary
            with open(self.data_dir / "data_summary.pkl", 'rb') as f:
                summary = pickle.load(f)
            
            print("✅ Loaded previously processed data")
            self._print_summary(summary)
            
            return X, y, summary
            
        except FileNotFoundError as e:
            print(f"❌ No processed data found: {e}")
            print("💡 Run full pipeline first with: pipeline.run_full_pipeline()")
            return None, None, None
    
    def save_training_data(self, X: np.ndarray, y: np.ndarray):
        """Save training data for later use."""
        save_dir = self.data_dir / "preprocessed"
        save_dir.mkdir(exist_ok=True)
        
        np.save(save_dir / "X_train.npy", X)
        np.save(save_dir / "y_train.npy", y)
        
        print(f"💾 Training data saved to {save_dir}/")
        print(f"   - X_train.npy: {X.shape}")
        print(f"   - y_train.npy: {y.shape}")

# Quick usage function
def prepare_data(num_books: int = 3, sequence_length: int = 100, data_dir: str = "data"):
    """Quick function to prepare data for training."""
    print("🎯 Quick Data Preparation")
    print("-" * 30)
    
    pipeline = DataPipeline(data_dir=data_dir, sequence_length=sequence_length)
    X, y, summary = pipeline.run_full_pipeline(num_books=num_books)
    pipeline.save_training_data(X, y)
    
    return X, y, summary

# Example usage
if __name__ == "__main__":
    print("🧪 Testing DataPipeline...")
    
    # Test the complete pipeline
    pipeline = DataPipeline(sequence_length=50)  # Smaller sequence for testing
    X, y, summary = pipeline.run_full_pipeline(num_books=2)
    
    print("\n🎉 Pipeline test completed!")
    print(f"✅ Ready for training with {len(X):,} sequences")
    print(f"📊 Input shape: {X.shape}")
    print(f"🎯 Target shape: {y.shape}")
