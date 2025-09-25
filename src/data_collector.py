"""
Data collection module for downloading and processing Project Gutenberg texts.
Located in: src/data_collector.py
"""
import requests
import os
import re
import csv
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from bs4 import BeautifulSoup
import time

class GutenbergDataCollector:
    """Download and collect text data from Project Gutenberg."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Base URLs
        self.base_url = "https://www.gutenberg.org/files"
        self.catalog_url = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
        
    def download_catalog(self) -> pd.DataFrame:
        """Download the Project Gutenberg catalog."""
        print("📥 Downloading Project Gutenberg catalog...")
        try:
            response = requests.get(self.catalog_url, timeout=30)
            response.raise_for_status()
            
            # Save catalog
            catalog_path = self.raw_dir / "pg_catalog.csv"
            with open(catalog_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # Load as DataFrame
            df = pd.read_csv(catalog_path)
            print(f"✅ Catalog downloaded: {len(df)} books available")
            return df
            
        except Exception as e:
            print(f"❌ Error downloading catalog: {e}")
            return pd.DataFrame()
    
    def filter_english_books(self, df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
        """Filter for English books suitable for text generation."""
        if df.empty:
            return df
            
        # Filter for English books
        english_books = df[df['Language'].str.contains('en', na=False, case=False)]
        
        # Filter for books with reasonable length (exclude very short texts)
        # Focus on literature, fiction, and classic texts
        genres_keywords = ['fiction', 'literature', 'novel', 'story', 'tales', 'classic']
        genre_filter = english_books['Subjects'].str.contains(
            '|'.join(genres_keywords), na=False, case=False
        )
        
        filtered_books = english_books[genre_filter].head(limit)
        print(f"📚 Filtered to {len(filtered_books)} English literature books")
        return filtered_books
    
    def download_book_text(self, book_id: int) -> Optional[str]:
        """Download text content for a specific book ID."""
        # Try different text formats
        text_urls = [
            f"{self.base_url}/{book_id}/{book_id}-0.txt",  # UTF-8
            f"{self.base_url}/{book_id}/{book_id}.txt",    # Plain text
            f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"  # Alternative
        ]
        
        for url in text_urls:
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    # Try to decode properly
                    try:
                        text = response.content.decode('utf-8')
                    except UnicodeDecodeError:
                        text = response.content.decode('latin-1')
                    
                    print(f"✅ Downloaded book {book_id} ({len(text):,} characters)")
                    return text
                    
            except Exception as e:
                continue
        
        print(f"❌ Failed to download book {book_id}")
        return None
    
    def clean_gutenberg_text(self, text: str) -> str:
        """Clean Project Gutenberg text by removing headers and footers."""
        if not text:
            return ""
        
        lines = text.split('\n')
        
        # Find start of actual content (after Gutenberg header)
        start_idx = 0
        for i, line in enumerate(lines):
            if any(marker in line.upper() for marker in [
                'START OF THE PROJECT GUTENBERG',
                'START OF THIS PROJECT GUTENBERG',
                '*** START OF'
            ]):
                start_idx = i + 1
                break
        
        # Find end of actual content (before Gutenberg footer)
        end_idx = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if any(marker in lines[i].upper() for marker in [
                'END OF THE PROJECT GUTENBERG',
                'END OF THIS PROJECT GUTENBERG',
                '*** END OF'
            ]):
                end_idx = i
                break
        
        # Extract main content
        content_lines = lines[start_idx:end_idx]
        clean_text = '\n'.join(content_lines)
        
        # Basic cleaning
        clean_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', clean_text)  # Multiple newlines
        clean_text = re.sub(r'[ \t]+', ' ', clean_text)  # Multiple spaces
        clean_text = clean_text.strip()
        
        return clean_text
    
    def collect_books(self, num_books: int = 5) -> List[str]:
        """Collect and clean multiple books."""
        print(f"🚀 Starting collection of {num_books} books...")
        
        # Download catalog
        catalog = self.download_catalog()
        if catalog.empty:
            print("❌ No catalog available, using sample data")
            return []
        
        # Filter books
        filtered_books = self.filter_english_books(catalog, limit=num_books * 3)
        
        if filtered_books.empty:
            print("❌ No suitable books found")
            return []
        
        collected_texts = []
        successful_downloads = 0
        
        for _, book_row in filtered_books.iterrows():
            if successful_downloads >= num_books:
                break
                
            book_id = book_row['Text#']
            title = book_row.get('Title', f'Book_{book_id}')
            
            print(f"📖 Processing: {title} (ID: {book_id})")
            
            # Download text
            raw_text = self.download_book_text(book_id)
            if raw_text:
                # Clean text
                clean_text = self.clean_gutenberg_text(raw_text)
                
                if len(clean_text) > 10000:  # Minimum length check
                    # Save individual book
                    book_filename = f"book_{book_id}_{title[:30]}.txt".replace(' ', '_')
                    book_filename = re.sub(r'[^\w_.-]', '', book_filename)
                    book_path = self.processed_dir / book_filename
                    
                    with open(book_path, 'w', encoding='utf-8') as f:
                        f.write(clean_text)
                    
                    collected_texts.append(clean_text)
                    successful_downloads += 1
                    print(f"✅ Successfully processed: {title}")
                else:
                    print(f"⚠️ Book too short, skipping: {title}")
            
            # Be respectful to the server
            time.sleep(1)
        
        print(f"🎉 Collection complete: {len(collected_texts)} books collected")
        return collected_texts

# Example usage
if __name__ == "__main__":
    collector = GutenbergDataCollector()
    texts = collector.collect_books(num_books=3)  # Start with 3 books for testing
    print(f"📚 Total texts collected: {len(texts)}")
