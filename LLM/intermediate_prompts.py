"""
Optimized Intermediate Prompt Database Creation Class
Creates prompts with relevant chunks based on similarity search
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from collections import defaultdict
import warnings
import gc,torch
warnings.filterwarnings('ignore')


class OptimizedPromptCreator:
    """
    Optimized version for creating intermediate prompt database by finding relevant chunks
    based on similarity to queries/prompts
    """
    
    def __init__(self, batch_size: int = 1000):
        """
        Initialize optimized prompt creator
        
        Args:
            batch_size: Batch size for vectorized operations
        """
        self.model = None
        self.vector_db = None
        self.batch_size = batch_size
        
    def _load_embedding_model(self, model_name: str) -> SentenceTransformer:
        """
        Load embedding model by name with error handling
        
        Args:
            model_name: Name of the embedding model to load
            
        Returns:
            Loaded SentenceTransformer model
        """
        print(f"Loading embedding model: {model_name}")
        
        # Map common model names to sentence-transformers model names
        model_mapping = {
            "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
            "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
            "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
            "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2"
        }
        
        if model_name in model_mapping:
            model_name = model_mapping[model_name]
        
        try:
            self.model = SentenceTransformer(model_name)
            print(f"Model loaded successfully.")
            return self.model
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise
    
    def _load_vector_database(self, vector_db_path: str) -> Dict[str, Any]:
        """Load existing vector database with optimized memory usage"""
        print(f"Loading vector database from: {vector_db_path}")
        data = np.load(vector_db_path)
        
        # Store frequently accessed data as numpy arrays for faster access
        self.vector_db = {
            'embeddings': data['embeddings'],
            'model_name': str(data['model_name']),
            'embedding_dim': int(data['embedding_dim']),
            'num_chunks': int(data['num_chunks']),
            'dois': data['dois'],  # Keep as numpy array for vectorized operations
            'chunk_indices': data['chunk_indices']  # Keep as numpy array
        }
        
        print(f"Vector database loaded: {self.vector_db['num_chunks']} chunks")
        return self.vector_db
    
    def _load_text_database(self, csv_path: str) -> pd.DataFrame:
        """Load text database from CSV with optimized data types"""
        print(f"Loading text database from: {csv_path}")
        if ".csv" in csv_path:
            df = pd.read_csv(csv_path)
        elif ".pkl" in csv_path:
            df = pd.read_pickle(csv_path)
        else:
            raise ValueError(f"Unsupported file format: {csv_path}")
        
        # Optimize DataFrame for faster operations
        if 'chunk_text' in df.columns:
            df['chunk_text'] = df['chunk_text'].fillna('').astype(str)
        if 'doi' in df.columns:
            df['doi'] = df['doi'].astype(str)
        if 'chunk_index' in df.columns:
            df['chunk_index'] = df['chunk_index'].astype(np.int32)
            
        print(f"Text database loaded: {len(df)} chunks")
        return df
    
    def _create_chunk_lookup(self, text_db: pd.DataFrame) -> Dict[Tuple[str, int], str]:
        """
        Create optimized chunk text lookup dictionary using vectorized operations
        
        Args:
            text_db: DataFrame with text chunks
            
        Returns:
            Dictionary mapping (doi, chunk_index) to chunk text
        """
        # Use vectorized operations instead of iterrows()
        chunk_texts = {}
        
        # Create unique keys for each row using vectorized operations
        keys = list(zip(text_db['doi'].values, text_db['chunk_index'].values))
        
        # Create dictionary using list comprehension (much faster than iterrows)
        chunk_texts = {key: str(text) for key, text in 
                      zip(keys, text_db['chunk_text'].values)}
        
        return chunk_texts
    
    def _find_similar_chunks_vectorized(self, query_embedding: np.ndarray, 
                                      threshold: float = 0.5, 
                                      top_k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimized chunk finding using vectorized operations
        
        Args:
            query_embedding: Embedding vector of the query
            threshold: Minimum similarity threshold
            top_k: Maximum number of chunks to return (None for all above threshold)
            
        Returns:
            Tuple of (indices, similarities) for similar chunks
        """
        # Vectorized similarity calculation
        similarities = cosine_similarity([query_embedding], self.vector_db['embeddings'])[0]
        
        # Find indices above threshold using boolean indexing
        valid_mask = similarities >= threshold
        valid_indices = np.where(valid_mask)[0]
        valid_similarities = similarities[valid_indices]
        
        # Sort by similarity (descending) - vectorized
        if len(valid_indices) > 0:
            sort_order = np.argsort(valid_similarities)[::-1]
            sorted_indices = valid_indices[sort_order]
            sorted_similarities = valid_similarities[sort_order]
            
            # Apply top_k limit if specified
            if top_k is not None and len(sorted_indices) > top_k:
                sorted_indices = sorted_indices[:top_k]
                sorted_similarities = sorted_similarities[:top_k]
                
            return sorted_indices, sorted_similarities
        else:
            return np.array([]), np.array([])
    
    def _combine_chunks_optimized(self, chunk_info_list: List[Dict], 
                                chunk_texts: Dict[Tuple[str, int], str], 
                                max_chars: int = 12000) -> str:
        """
        Optimized chunk combination using efficient string building
        
        Args:
            chunk_info_list: List of chunk information
            chunk_texts: Dictionary mapping (doi, chunk_index) to chunk text
            max_chars: Maximum total characters to include
            
        Returns:
            Combined text from selected chunks
        """
        # Pre-allocate list for better performance
        text_parts = []
        current_length = 0
        
        for chunk_info in chunk_info_list:
            key = (chunk_info['doi'], chunk_info['chunk_index'])
            
            # Skip if chunk text not found
            if key not in chunk_texts:
                continue
                
            chunk_text = chunk_texts[key]
            
            # Check if adding this chunk would exceed limit
            if current_length + len(chunk_text) > max_chars:
                remaining_chars = max_chars - current_length
                if remaining_chars > 100:  # Only add if meaningful text remains
                    truncated_text = chunk_text[:remaining_chars].strip()
                    if truncated_text:
                        text_parts.append(truncated_text)
                break
            
            text_parts.append(chunk_text)
            current_length += len(chunk_text) + 2  # +2 for "\n\n"
        
        # Join with newlines for efficient string building
        return "\n\n".join(text_parts).strip()
    
    def create_intermediate_prompts(self, 
                                  model_name: str,
                                  vector_db_path: str,
                                  text_db_path: str,
                                  initial_prompt: str,
                                    template: str,
                                  threshold: float = 0.5,
                                  max_chars_per_article: int = 12000,
                                  top_k_chunks: int = None) -> pd.DataFrame:
        """
        Create intermediate prompt database using optimized similarity-based chunk selection
        
        Args:
            model_name: Name of the embedding model
            vector_db_path: Path to vector database (.npz file)
            text_db_path: Path to text database (CSV file)
            initial_prompt: Initial prompt text to use for similarity search
            template: A text that would be embedded and used to search for relevant context
            threshold: Similarity threshold for chunk selection
            max_chars_per_article: Maximum characters per article to include
            top_k_chunks: Maximum number of chunks to consider per article (None for all)
            
        Returns:
            DataFrame with columns: doi, combined_prompt
        """
        print("Loading components...")
        
        # Load components
        self._load_embedding_model(model_name)
        self._load_vector_database(vector_db_path)
        text_db = self._load_text_database(text_db_path)
        
        print("Creating optimized chunk lookup...")
        # Create optimized chunk text lookup
        chunk_texts = self._create_chunk_lookup(text_db)
        
        print("Embedding search template...")
        # Step 1: Embed the initial prompt only once
        search_template_embedding = self.model.encode([template], 
                                                   normalize_embeddings=True)[0]
        
        print("Finding similar chunks...")
        # Step 2: Find similar chunks using vectorized operations
        similar_indices, similarities = self._find_similar_chunks_vectorized(
            search_template_embedding, threshold, top_k_chunks)
        
        if len(similar_indices) == 0:
            print("No similar chunks found above threshold.")
            return pd.DataFrame(columns=['doi', 'combined_prompt'])
        
        print(f"Found {len(similar_indices)} similar chunks")
        
        # Step 3: Vectorized chunk information creation
        chunk_infos = []
        dois = self.vector_db['dois']
        chunk_indices = self.vector_db['chunk_indices']
        
        for idx in range(len(similar_indices)):
            chunk_idx = similar_indices[idx]
            chunk_infos.append({
                'doi': str(dois[chunk_idx]),
                'chunk_index': int(chunk_indices[chunk_idx]),
                'similarity': float(similarities[idx])
            })
        
        print("Grouping chunks by DOI...")
        # Step 4: Efficient grouping using defaultdict
        doi_to_chunks = defaultdict(list)
        
        for chunk_info in chunk_infos:
            doi_to_chunks[chunk_info['doi']].append(chunk_info)
        
        # Sort chunks within each DOI by similarity (vectorized)
        #for doi in doi_to_chunks:
            #doi_to_chunks[doi].sort(key=lambda x: x['similarity'], reverse=True)
        
        print(f"Processing {len(doi_to_chunks)} unique DOIs...")
        # Step 5: Create combined prompts for each DOI
        results = []
        
        for doi, chunks in doi_to_chunks.items():
            # Use optimized chunk combination
            combined_text = self._combine_chunks_optimized(chunks, chunk_texts, max_chars_per_article)
            
            if not combined_text:
                continue
            
            # Create the final prompt
            final_prompt = f"{initial_prompt}\n\n**Context:**\n{combined_text}\n\n**Answer:"
            
            results.append({
                'doi': doi,
                'combined_prompt': final_prompt
            })
        
        print("Creating DataFrame and calculating statistics...")
        # Create DataFrame
        df = pd.DataFrame(results)
        
        if len(df) > 0:
            print(f"Summary statistics:")
            context_lengths = df['combined_prompt'].str.len()
            print(f"  - Mean prompt length: {context_lengths.mean():.0f} characters")
            print(f"  - Min prompt length: {context_lengths.min():.0f} characters")
            print(f"  - Max prompt length: {context_lengths.max():.0f} characters")
            print(f"  - Total DOIs processed: {len(df)}")
        else:
            print("No results generated.")
        
        return df
    
    def save_to_csv(self, df: pd.DataFrame, output_path: str) -> None:
        """Save DataFrame to CSV file with optimized settings"""
        # Use efficient CSV writing with compression if output is large
        if len(df) > 10000:
            df.to_csv(output_path, index=False, compression='gzip')
            print(f"Intermediate prompt database saved to: {output_path} (compressed)")
        else:
            df.to_csv(output_path, index=False)
            print(f"Intermediate prompt database saved to: {output_path}")


def main(model, vector_db_path, text_db_path, prompt, threshold, max_chars, top_k, out):
    """Example usage with performance monitoring"""
    
    # Create intermediate prompts
    creator = OptimizedPromptCreator(batch_size=1000)
    with open(prompt) as f:
        initial_prompt = f.read()
    if top_k == 0:
        top_k = None
    try:
        intermediate_df = creator.create_intermediate_prompts(
            model_name=model,
            vector_db_path=vector_db_path,
            text_db_path=text_db_path,
            initial_prompt=initial_prompt,
            threshold=threshold,
            max_chars_per_article=max_chars,
            top_k_chunks=top_k
        )
        
        # Save to pickle
        intermediate_df.to_pickle(f"{out}.pkl.gz")
        del intermediate_df, creator
    except Exception as e:
        print(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intermediate Prompt Set Creation Tool")
    parser.add_argument("--prompt", type=str, required=True, help="Path to the txt file of prompt template")
    parser.add_argument("--model", type=str, default="bge-large-en-v1.5", help="Name of embedding model")
    parser.add_argument("--vector_db_path", type=str, required=True, help="Path to vector database file")
    parser.add_argument("--text_db_path", type=str, required=True, help="Path to text database file")
    parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold for vector comparisons")
    parser.add_argument("--max_chars", type=int, default=9000, help="Maximum character count for context")
    parser.add_argument("--topk", type=int, default=0, help="Choose top k chunks as context")
    parser.add_argument("--out", type=str, default="intermediate_database", help="Name of output")

    args = parser.parse_args()

    try:
        main(args.model, args.vector_db_path, args.text_db_path, args.prompt, args.threshold, args.max_chars, args.topk, args.out)
        torch.cuda.empty_cache()
        gc.collect()
    except KeyboardInterrupt:
        print("Process interrupted by user. Exiting...")
    finally:
        torch.cuda.empty_cache()
        gc.collect()
