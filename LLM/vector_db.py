"""
Enhanced Vector Database Creator with Multi-GPU Parallel Processing
Creates embeddings for text chunks across multiple GPUs while maintaining original order
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import os
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import logging
from dataclasses import dataclass
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GPUDevice:
    """Represents a GPU device with its properties"""
    id: int
    name: str
    memory_total: int
    memory_free: int

class MultiGPUEmbeddingProcessor:
    """
    Multi-GPU embedding processor that distributes work across available GPUs
    while maintaining the original order of text chunks
    """
    
    def __init__(self, gpu_memory_threshold: float = 0.8, progress_gpu_id: int = 0):
        """
        Initialize the multi-GPU processor
        
        Args:
            gpu_memory_threshold: Maximum memory usage threshold (0.0-1.0)
            progress_gpu_id: GPU ID to show progress bar for (default: 0)
        """
        self.gpu_memory_threshold = gpu_memory_threshold
        self.progress_gpu_id = progress_gpu_id
        self.models = {}
        self.device_info = []
        self._setup_gpu_devices()
        
    def _setup_gpu_devices(self):
        """Detect and setup available GPU devices"""
        if not torch.cuda.is_available():
            logger.warning("No CUDA devices available. Falling back to CPU processing.")
            return
            
        num_gpus = torch.cuda.device_count()
        logger.info(f"Detected {num_gpus} CUDA devices")
        
        for i in range(num_gpus):
            device_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_free = memory_total - memory_allocated
            
            device = GPUDevice(
                id=i,
                name=device_name,
                memory_total=memory_total,
                memory_free=memory_free
            )
            
            self.device_info.append(device)
            logger.info(f"GPU {i}: {device_name}, Free: {memory_free//1024**2}MB / {memory_total//1024**2}MB")
            
    def _load_model_on_device(self, model_name: str, device_id: int) -> SentenceTransformer:
        """
        Load embedding model on specific GPU device
        
        Args:
            model_name: Name of the embedding model to load
            device_id: GPU device ID
            
        Returns:
            Loaded SentenceTransformer model
        """
        device = f"cuda:{device_id}"
        logger.info(f"Loading model {model_name} on device {device}")
        
        # Map common model names to sentence-transformers model names
        model_mapping = {
            "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
            "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5", 
            "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
            "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2"
        }
        
        model_name_mapped = model_mapping.get(model_name, model_name)
        
        try:
            model = SentenceTransformer(model_name_mapped)
            model.to(device)
            model.eval()
            
            # Warm up the model
            test_embedding = model.encode(["warm up"], device=device)
            
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name} on device {device}: {e}")
            raise
            
    def _create_embeddings_batch(self, texts: List[str], model: SentenceTransformer, 
                               device: str, batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """
        Create embeddings for a batch of texts on a specific device
        
        Args:
            texts: List of text strings to embed
            model: SentenceTransformer model
            device: Device string (e.g., 'cuda:0')
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar for this device
            
        Returns:
            Numpy array of embeddings
        """
        logger.info(f"Creating embeddings for {len(texts)} texts on {device}")
        
        all_embeddings = []
        
        # Only show progress bar for GPU 0 (or CPU when no GPUs available)
        show_pbar = show_progress and (
            device == "cuda:0" or 
            (device.startswith("cuda:") == False)  # CPU devices
        )
        
        for i in tqdm(range(0, len(texts), batch_size), 
                      desc=f"Processing batches on {device}" if show_pbar else None,
                      disable=not show_pbar):
            batch = texts[i:i + batch_size]
            batch_embeddings = model.encode(
                batch, 
                device=device,
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=len(batch)  # Process entire batch at once
            )
            all_embeddings.append(batch_embeddings)
            
            # Clear GPU cache periodically
            if i % (batch_size * 5) == 0:
                torch.cuda.empty_cache()
        
        # Concatenate all batches
        final_embeddings = np.vstack(all_embeddings)
        logger.info(f"Embeddings created on {device}. Shape: {final_embeddings.shape}")
        
        return final_embeddings
        
    def _distribute_chunks(self, chunk_texts: List[str], num_gpus: int) -> List[List[int]]:
        """
        Distribute chunk indices across available GPUs in a balanced way
        
        Args:
            chunk_texts: List of text chunks
            num_gpus: Number of available GPUs
            
        Returns:
            List of index lists, one per GPU
        """
        total_chunks = len(chunk_texts)
        chunks_per_gpu = total_chunks // num_gpus
        remaining_chunks = total_chunks % num_gpus
        
        distributions = []
        start_idx = 0
        
        for i in range(num_gpus):
            # Distribute remaining chunks to first few GPUs
            current_chunk_count = chunks_per_gpu + (1 if i < remaining_chunks else 0)
            
            if current_chunk_count > 0:
                end_idx = start_idx + current_chunk_count
                gpu_indices = list(range(start_idx, end_idx))
                distributions.append(gpu_indices)
                start_idx = end_idx
            else:
                distributions.append([])
                
        return distributions
        
    def create_embeddings_parallel(self, model_name: str, chunk_texts: List[str],
                                 batch_size: int = 64) -> Tuple[np.ndarray, List[int]]:
        """
        Create embeddings in parallel across multiple GPUs
        
        Args:
            model_name: Name of the embedding model to use
            chunk_texts: List of text chunks to embed
            batch_size: Batch size for each GPU processing
            
        Returns:
            Tuple of (embeddings_array, original_indices)
        """
        if not torch.cuda.is_available() or len(self.device_info) == 0:
            logger.warning("Using CPU processing (no GPUs available)")
            return self._create_embeddings_cpu(model_name, chunk_texts, batch_size)
            
        available_gpus = [gpu for gpu in self.device_info if gpu.memory_free > 1024**3]  # At least 1GB free
        
        if not available_gpus:
            logger.warning("No GPUs with sufficient memory. Using CPU processing")
            return self._create_embeddings_cpu(model_name, chunk_texts, batch_size)
            
        logger.info(f"Using {len(available_gpus)} GPUs for parallel processing")
        
        # Distribute chunks across GPUs
        gpu_distributions = self._distribute_chunks(chunk_texts, len(available_gpus))
        
        # Load models on each GPU
        for gpu in available_gpus:
            try:
                model = self._load_model_on_device(model_name, gpu.id)
                self.models[gpu.id] = model
            except Exception as e:
                logger.error(f"Failed to load model on GPU {gpu.id}: {e}")
                available_gpus.remove(gpu)
                
        if not available_gpus:
            logger.warning("No GPUs could load the model. Falling back to CPU")
            return self._create_embeddings_cpu(model_name, chunk_texts, batch_size)
        
        # Process embeddings in parallel
        all_embeddings = []
        all_original_indices = []
        
        with ThreadPoolExecutor(max_workers=len(available_gpus)) as executor:
            futures = []
            
            for gpu, indices in zip(available_gpus, gpu_distributions):
                if indices:  # Only process if this GPU has chunks
                    gpu_texts = [chunk_texts[i] for i in indices]
                    model = self.models[gpu.id]
                    
                    # Only show progress bar for the configured GPU
                    show_progress = (gpu.id == self.progress_gpu_id)
                    
                    future = executor.submit(
                        self._create_embeddings_batch,
                        gpu_texts,
                        model,
                        f"cuda:{gpu.id}",
                        batch_size,
                        show_progress
                    )
                    futures.append((future, indices))
            
            # Collect results while maintaining order
            results = []
            for future, original_indices in futures:
                try:
                    embeddings = future.result()
                    results.append((embeddings, original_indices))
                    logger.info(f"Completed processing {len(original_indices)} chunks")
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    raise
        
        # Sort results by original indices to maintain order
        results.sort(key=lambda x: min(x[1]))
        
        # Reconstruct final arrays
        final_embeddings = []
        final_indices = []
        
        for embeddings, indices in results:
            final_embeddings.append(embeddings)
            final_indices.extend(indices)
            
        # Concatenate all embeddings
        final_embeddings_array = np.vstack(final_embeddings)
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        
        logger.info(f"Parallel processing complete. Final shape: {final_embeddings_array.shape}")
        
        return final_embeddings_array, final_indices
        
    def _create_embeddings_cpu(self, model_name: str, chunk_texts: List[str],
                              batch_size: int = 64) -> Tuple[np.ndarray, List[int]]:
        """Fallback CPU processing"""
        logger.info("Processing embeddings on CPU")
        
        # Map model names
        model_mapping = {
            "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
            "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
            "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5", 
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
            "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2"
        }
        
        model_name_mapped = model_mapping.get(model_name, model_name)
        model = SentenceTransformer(model_name_mapped)
        
        # Process in batches
        all_embeddings = []
        indices = list(range(len(chunk_texts)))
        
        for i in tqdm(range(0, len(chunk_texts), batch_size), desc="CPU Processing"):
            batch = chunk_texts[i:i + batch_size]
            batch_embeddings = model.encode(
                batch, 
                show_progress_bar=False, 
                normalize_embeddings=True
            )
            all_embeddings.append(batch_embeddings)
            
        final_embeddings_array = np.vstack(all_embeddings)
        
        return final_embeddings_array, indices


class EnhancedVectorDatabaseCreator:
    """
    Enhanced vector database creator with multi-GPU support
    """
    
    def __init__(self, use_multi_gpu: bool = True, gpu_memory_threshold: float = 0.8, progress_gpu_id: int = 0):
        """
        Initialize enhanced vector database creator
        
        Args:
            use_multi_gpu: Whether to use multi-GPU processing
            gpu_memory_threshold: GPU memory usage threshold
            progress_gpu_id: GPU ID to show progress bar for (default: 0)
        """
        self.use_multi_gpu = use_multi_gpu and torch.cuda.is_available()
        self.embedding_processor = MultiGPUEmbeddingProcessor(
            gpu_memory_threshold=gpu_memory_threshold,
            progress_gpu_id=progress_gpu_id
        )
        self.model = None
        self.embedding_dim = None
        
    def _load_embedding_model(self, model_name: str) -> SentenceTransformer:
        """Load embedding model (single GPU/CPU version for compatibility)"""
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
            if self.use_multi_gpu:
                # For multi-GPU, we'll load on CPU first to get embedding dimension
                self.model = SentenceTransformer(model_name)
            else:
                self.model = SentenceTransformer(model_name)
                
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            print(f"Multi-GPU processing: {'Enabled' if self.use_multi_gpu else 'Disabled'}")
            return self.model
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise
    
    def create_vector_database(self, model_name: str, csv_path: str, 
                             output_dir: str = ".", batch_size: int = 64, 
                               text_col: str = 'chunk_text', chunk_idx: str = 'chunk_index', 
                              doi_col: str = 'doi') -> str:
        """
        Create vector database from CSV file with optional multi-GPU processing
        
        Args:
            model_name: Name of the embedding model to use
            csv_path: Path to the CSV file from PDF extraction
            output_dir: Directory to save the vector database
            batch_size: Batch size for embedding processing
            
        Returns:
            Path to the created vector database file
        """
        # Load CSV data
        print(f"Loading text database from: {csv_path}")
        if ".csv" in csv_path:
            df = pd.read_csv(csv_path)
        elif ".pkl" in csv_path:
            df = pd.read_pickle(csv_path)
        else:
            raise ValueError(f"Unsupported file format: {csv_path}")
            
        print(f"Loaded {len(df)} chunks")

        assert len({text_col, chunk_idx, doi_col}.intersection(set(df.columns))) == 3, 'Provided columns not found in provided dataframe.'
        # Load embedding model
        self._load_embedding_model(model_name)
        
        # Extract chunk texts and maintain original order
        chunk_texts = df[text_col].tolist()
        
        # Create embeddings using multi-GPU or single GPU/CPU
        if self.use_multi_gpu:
            print("Using multi-GPU parallel processing...")
            embeddings, original_indices = self.embedding_processor.create_embeddings_parallel(
                model_name, chunk_texts, batch_size
            )
            
            # Verify order is maintained
            if original_indices != list(range(len(chunk_texts))):
                logger.warning("Reordering embeddings to match original chunk order")
                # Reorder embeddings based on original indices
                reordered_embeddings = np.zeros_like(embeddings)
                for i, orig_idx in enumerate(original_indices):
                    reordered_embeddings[orig_idx] = embeddings[i]
                embeddings = reordered_embeddings
                
        else:
            print("Using single GPU/CPU processing...")
            embeddings = self._create_embeddings_single(chunk_texts, self.model, batch_size)
        
        # Prepare metadata
        metadata = {
            'model_name': model_name,
            'embedding_dim': self.embedding_dim,
            'num_chunks': len(df),
            'dois': df[doi_col].tolist(),
            'chunk_indices': df[chunk_idx].tolist(),
            'multi_gpu_used': self.use_multi_gpu,
            'device_info': [gpu.name for gpu in self.embedding_processor.device_info] if self.use_multi_gpu else ['CPU']
        }
        
        # Create output filename
        csv_filename = os.path.basename(csv_path).replace('.csv', '').replace('.pkl', '')
        gpu_suffix = "_multi_gpu" if self.use_multi_gpu else "_single_gpu"
        output_filename = f"{csv_filename}_embeddings_{model_name.replace('/', '_')}{gpu_suffix}.npz"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save to NPZ file
        print(f"Saving vector database to: {output_path}")
        np.savez_compressed(
            output_path,
            embeddings=embeddings,
            model_name=metadata['model_name'],
            embedding_dim=metadata['embedding_dim'],
            num_chunks=metadata['num_chunks'],
            dois=np.array(metadata['dois']),
            chunk_indices=np.array(metadata['chunk_indices']),
            multi_gpu_used=np.array(metadata['multi_gpu_used']),
            device_info=np.array(metadata['device_info'], dtype=object)
        )
        
        print(f"Vector database created successfully!")
        print(f"  - Embeddings shape: {embeddings.shape}")
        print(f"  - Model: {model_name}")
        print(f"  - Total chunks: {metadata['num_chunks']}")
        print(f"  - Multi-GPU used: {metadata['multi_gpu_used']}")
        print(f"  - Devices: {metadata['device_info']}")
        
        return output_path
    
    def _create_embeddings_single(self, texts: List[str], model: SentenceTransformer, 
                                 batch_size: int = 64) -> np.ndarray:
        """
        Create embeddings for list of texts (single device)
        
        Args:
            texts: List of text strings to embed
            model: SentenceTransformer model
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings
        """
        print(f"Creating embeddings for {len(texts)} texts...")
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            batch_embeddings = model.encode(
                batch, 
                show_progress_bar=False, 
                normalize_embeddings=True
            )
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        final_embeddings = np.vstack(all_embeddings)
        print(f"Embeddings created. Shape: {final_embeddings.shape}")
        
        return final_embeddings
    
    def load_vector_database(self, npz_path: str) -> Dict[str, Any]:
        """
        Load existing vector database
        
        Args:
            npz_path: Path to the .npz file
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        data = np.load(npz_path, allow_pickle=True)
        
        return {
            'embeddings': data['embeddings'],
            'model_name': str(data['model_name']),
            'embedding_dim': int(data['embedding_dim']),
            'num_chunks': int(data['num_chunks']),
            'dois': data['dois'].tolist(),
            'chunk_indices': data['chunk_indices'].tolist(),
            'multi_gpu_used': bool(data.get('multi_gpu_used', False)),
            'device_info': data.get('device_info', ['Unknown']).tolist() if 'device_info' in data else ['Unknown']
        }


def benchmark_processing(model_name: str, chunk_texts: List[str], 
                        batch_sizes: List[int] = [32, 64, 128]) -> Dict[str, float]:
    """
    Benchmark different batch sizes for performance optimization
    
    Args:
        model_name: Name of the embedding model
        chunk_texts: List of text chunks
        batch_sizes: List of batch sizes to test
        
    Returns:
        Dictionary mapping batch sizes to processing times
    """
    import time
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch size: {batch_size}")
        
        processor = EnhancedVectorDatabaseCreator(use_multi_gpu=True)
        processor._load_embedding_model(model_name)
        
        start_time = time.time()
        
        # Use a subset for benchmark
        subset_size = min(1000, len(chunk_texts))
        test_texts = chunk_texts[:subset_size]
        
        embeddings, _ = processor.embedding_processor.create_embeddings_parallel(
            model_name, test_texts, batch_size
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        chunks_per_second = subset_size / processing_time
        
        results[batch_size] = processing_time
        print(f"Batch size {batch_size}: {processing_time:.2f}s ({chunks_per_second:.2f} chunks/sec)")
    
    return results


def main():
    """Example usage with multi-GPU support"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-GPU Vector Database Creator')
    parser.add_argument('--model', default='bge-large-en-v1.5', help='Embedding model name')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', default='.', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for processing')
    parser.add_argument('--single-gpu', action='store_true', help='Force single GPU processing')
    parser.add_argument('--benchmark', action='store_true', help='Run batch size benchmark')
    
    args = parser.parse_args()
    
    # Create vector database with multi-GPU support
    creator = EnhancedVectorDatabaseCreator(
        use_multi_gpu=not args.single_gpu,
        gpu_memory_threshold=0.8
    )
    
    if args.benchmark:
        # Load sample data for benchmarking
        df = pd.read_pickle(args.input) if ".pkl" in args.input else pd.read_csv(args.input)
        chunk_texts = df['chunk_text'].tolist()[:2000]  # Use subset for benchmark
        
        benchmark_results = benchmark_processing(args.model, chunk_texts)
        print("\nBenchmark Results:")
        for batch_size, time_taken in benchmark_results.items():
            print(f"Batch size {batch_size}: {time_taken:.2f}s")
        
        # Recommend best batch size
        best_batch_size = min(benchmark_results.keys(), key=lambda x: benchmark_results[x])
        print(f"\nRecommended batch size: {best_batch_size}")
        
        return
    
    # Create vector database
    vector_db_path = creator.create_vector_database(
        model_name=args.model,
        csv_path=args.input,
        output_dir=args.output,
        batch_size=args.batch_size
    )
    
    # Load and verify
    data = creator.load_vector_database(vector_db_path)
    print(f"\nLoaded vector database:")
    print(f"  - Model: {data['model_name']}")
    print(f"  - Embedding dimension: {data['embedding_dim']}")
    print(f"  - Number of chunks: {data['num_chunks']}")
    print(f"  - Multi-GPU used: {data['multi_gpu_used']}")
    print(f"  - Devices: {data['device_info']}")


if __name__ == "__main__":
    main()
