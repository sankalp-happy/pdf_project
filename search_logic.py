# search_logic.py
import os
import re
import time
import logging
from typing import List, Dict, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pdfplumber
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from google.api_core import retry, exceptions
import google.generativeai as genai
import nltk
nltk.download('punkt')  

# Configuration
DATA_DIR = "data"
PROCESSED_DIR = "processed"
EMBEDDINGS_DIR = "embeddings"
CHUNK_SIZE = 1000  # Characters
OVERLAP = 100      # Characters
API_RETRIES = 3

# Initialize logging
logging.basicConfig(filename='search.log', level=logging.INFO)

# Initialize Gemini
genai.configure(api_key="Enter_your_api_key")
model = genai.GenerativeModel('models/embedding-001')

class DocumentProcessor:
    def __init__(self, data_dir="data", processed_dir="processed", embeddings_dir="embeddings"):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.faiss_index = None
        self.tfidf_matrix = None
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    @retry.Retry(predicate=retry.if_exception_type(exceptions.ResourceExhausted))

    def get_embeddings(self, text: str) -> List[float]:
        if not isinstance(text, str) or not text.strip():
            logging.warning("Empty text input - returning dummy vector")
            return [0.0] * 768
        
        # Clean text before sending to API
        text = text.strip().replace("\x00", "")[:10000]  # Remove null bytes and truncate
        
        try:
            result = genai.embed_content(
                model='models/text-embedding-004',
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            return result['embedding']
        except Exception as e:
            logging.error(f"Embedding failed: {str(e)}")
            return [0.0] * 768

    def extract_metadata(self, file_path: str) -> Dict:
        """Extract metadata from PDF and filename"""
        metadata = {
            'title': os.path.basename(file_path),
            'author': 'Unknown',
            'year': datetime.now().year,
            'source': file_path
        }

        try:
            with pdfplumber.open(file_path) as pdf:
                # Extract PDF metadata
                meta = pdf.metadata
                metadata['title'] = meta.get('Title', metadata['title'])
                metadata['author'] = meta.get('Author', metadata['author'])
                metadata['year'] = meta.get('CreationDate', metadata['year'])[:4] if meta.get('CreationDate') else metadata['year']
                
                # Fallback to filename parsing
                if metadata['author'] == 'Unknown':
                    match = re.match(r"(.+)_(\d{4})_.+\.pdf", os.path.basename(file_path))
                    if match:
                        metadata['author'], metadata['year'] = match.groups()
        
        except Exception as e:
            logging.error(f"Metadata extraction failed for {file_path}: {str(e)}")
        
        return metadata

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks preserving sentence boundaries"""
        chunks = []
        sentences = sent_tokenize(text)
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > CHUNK_SIZE:
                chunks.append(" ".join(current_chunk))
                # Overlap by keeping last N sentences
                current_chunk = current_chunk[-int(OVERLAP/50):]  # Approximate overlap
                current_length = sum(len(s) for s in current_chunk)
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return [c.strip() for c in chunks if len(c.strip()) > 10]

    def process_documents(self):
        """Process all PDFs in data directory"""
        processed_path = self.processed_dir / "documents.csv"
        
        # Return cached data if valid
        if processed_path.exists():
            file_mtime = processed_path.stat().st_mtime
            data_files = [f.stat().st_mtime for f in Path(self.data_dir).glob("*.pdf")]
            
            # Check if any PDFs have changed since last processing
            if max(data_files, default=0) < file_mtime:
                logging.info("Using valid cached processed data")
                return pd.read_csv(processed_path)
        
        if (Path(self.processed_dir)/"documents.csv").exists():
            logging.info("Using cached processed data")
            return pd.read_csv(Path(self.processed_dir)/"documents.csv")
    
        processed_data = []
        
        files = list(Path(self.data_dir).glob("*.pdf"))
        if not files:
            logging.warning(f"No PDF files found in {self.data_dir}")
            return pd.DataFrame()

        logging.info(f"Processing {len(files)} files...")
        
        for i, file_path in enumerate(files, 1):
            try:
                logging.info(f"Processing {file_path.name} ({i}/{len(files)})")
                
                # Extract metadata
                metadata = self.extract_metadata(str(file_path))
                
                # Extract text
                with pdfplumber.open(file_path) as pdf:
                    text = "\n".join([
                        page.extract_text(x_tolerance=1, y_tolerance=1)
                        or f"<IMAGE PAGE {page.page_number}>" 
                        for page in pdf.pages
                    ])
    
                # Add fallback for completely image-based PDFs
                if "IMAGE PAGE" in text:
                    logging.warning(f"Image-based PDF detected: {file_path.name}")
                    text = "Document contains non-text elements"  # Fallback text

                    
                if not text.strip():
                    raise ValueError("No text extracted from PDF")
                    
                # Chunk text
                chunks = self.chunk_text(text)
                logging.info(f"Created {len(chunks)} chunks from {file_path.name}")
                
                # Add to processed data
                for chunk_idx, chunk in enumerate(chunks):
                    processed_data.append({
                        **metadata,
                        'chunk_id': f"{file_path.stem}_{chunk_idx}",
                        'text': chunk
                    })
                    
            except Exception as e:
                logging.error(f"Failed to process {file_path.name}: {str(e)}")
                continue
                # Save processed data
        if not processed_data:
            logging.error("No documents processed!")
            return pd.DataFrame()

        df = pd.DataFrame(processed_data)
        output_path = Path(self.processed_dir) / "documents.csv"
        df.to_csv(output_path, index=False)
        logging.info(f"Saved processed data to {output_path}")
        
        return df

    def build_indices(self):
        """Build FAISS and TF-IDF indices"""
        csv_path = Path(PROCESSED_DIR) / "documents.csv"
    
        if not csv_path.exists():
            logging.error("No processed documents CSV found!")
            return

        df = pd.read_csv(csv_path)
        if df.empty:
            logging.error("Processed documents CSV is empty!")
            return  # Add this check
        
        # TF-IDF Matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(df['text'])
        
        # FAISS Index
        embeddings = np.array([self.get_embeddings(text) for text in df['text']])
        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.faiss_index.add(embeddings)
        
        # Save indices
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        faiss.write_index(self.faiss_index, os.path.join(Path(EMBEDDINGS_DIR), f"faiss_{timestamp}.index"))
        pd.to_pickle(self.vectorizer, os.path.join(EMBEDDINGS_DIR, f"tfidf_{timestamp}.pkl"))

    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Perform hybrid semantic + keyword search"""
        # Semantic Search
        query_embedding = np.array([self.get_embeddings(query)])
        _, semantic_indices = self.faiss_index.search(query_embedding, top_k*2)
        semantic_indices = semantic_indices[0].astype('int64')

        # Keyword Search
        query_vec = self.vectorizer.transform([query])
        keyword_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        keyword_indices = keyword_scores.argsort()[-top_k*2:][::-1].astype('int64')

        # Combine results
        all_indices = set(semantic_indices).union(set(keyword_indices))
        results = []
        
        df = pd.read_csv(os.path.join(PROCESSED_DIR, "documents.csv"))
        
        for idx in all_indices:
            # Validate index bounds
            if idx < 0 or idx >= self.faiss_index.ntotal:
                continue
            
            try:
                # Convert to native Python int
                idx_int = int(idx)
                reconstructed_vec = self.faiss_index.reconstruct(idx_int)
                semantic_score = 1/(1 + np.linalg.norm(query_embedding - reconstructed_vec))
            except Exception as e:
                logging.error(f"Reconstruction error at index {idx}: {str(e)}")
                semantic_score = 0.0

            keyword_score = keyword_scores[idx]
            combined_score = self._calculate_combined_score(semantic_score, keyword_score, query)
            
            results.append({
                'title': df.iloc[idx]['title'],
                'author': df.iloc[idx]['author'],
                'year': df.iloc[idx]['year'],
                'semantic_score': semantic_score,
                'keyword_score': keyword_score,
                'score': combined_score,
                'snippet': df.iloc[idx]['text'][:200] + "...",
                'source': df.iloc[idx]['source']
            })

        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]

    def _calculate_combined_score(self, semantic: float, keyword: float, query: str) -> float:
        """Dynamically weight scores based on query characteristics"""
        query_length = len(query.split())
        unique_terms = len(set(query.lower().split()))
        
        # Long complex queries get more semantic weight
        if query_length > 7 or unique_terms > 5:
            return 0.7 * semantic + 0.3 * keyword
        # Short specific queries get balanced weighting
        elif query_length > 3:
            return 0.5 * semantic + 0.5 * keyword
        # Very short queries get more keyword weight
        else:
            return 0.3 * semantic + 0.7 * keyword