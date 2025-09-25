"""
Test script for the data pipeline
Located in: test_pipeline.py (root folder)
"""
import sys
sys.path.append('src')

from data_pipeline import prepare_data

if __name__ == "__main__":
    print("🧪 TESTING COMPLETE DATA PIPELINE")
    print("=" * 50)
    
    # Run pipeline with small dataset for testing
    try:
        X, y, summary = prepare_data(
            num_books=2,  # Small for testing
            sequence_length=50,  # Smaller sequence
            data_dir="data"
        )
        
        print(f"\n✅ SUCCESS! Training data ready:")
        print(f"   📊 Input sequences: {X.shape}")
        print(f"   🎯 Target sequences: {y.shape}")
        print(f"   🔤 Vocabulary size: {summary['vocab_size']:,}")
        print(f"   📚 Books processed: {summary['num_books']}")
        print("\n🎉 Ready for Step 3: Model Training!")
        
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        print("Please check your environment and dependencies")
