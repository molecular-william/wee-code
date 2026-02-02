# https://github.com/Zhaonan99/NVDT/blob/main/Part2-Feature%20extraction%20and%20standardization/features_dinucleotide.m
#https://www.nature.com/articles/s42003-022-03617-0#code-availability
import numpy as np
from numba import jit, prange
from typing import List, Tuple, Optional
import itertools

# ============================================================================
# Numba-compiled standalone functions (cannot be inside classes)
# ============================================================================

@jit(nopython=True, parallel=False, cache=True, fastmath=True)
def _compute_kmer_features_numba(encoded_seq: np.ndarray, N: int, k: int, 
                               position_offset: int, n_kmers: int) -> np.ndarray:
    """
    Numba-accelerated computation of k-mer features.
    This is a standalone function because Numba cannot compile class methods.
    
    Parameters:
    -----------
    encoded_seq : np.ndarray
        Integer-encoded DNA sequence
    N : int
        Length of the sequence
    k : int
        Size of k-mer
    position_offset : int
        Offset for position calculation: k*(k+1)/2
    n_kmers : int
        Total number of possible k-mers (4^k)
        
    Returns:
    --------
    np.ndarray of shape (3 * 4^k,) containing all features
    """
    # Initialize accumulators
    counts = np.zeros(n_kmers, dtype=np.int32)
    sum_pos = np.zeros(n_kmers, dtype=np.float64)
    sum_sq_pos = np.zeros(n_kmers, dtype=np.float64)
    
    # If sequence is shorter than k, return zeros
    if N < k:
        total_features = 3 * n_kmers
        return np.zeros(total_features, dtype=np.float64)
    
    # Process each k-mer in the sequence
    for i in range(N - k + 1):
        # Calculate k-mer index using sliding window
        idx = 0
        for j in range(k):
            idx = (idx << 2) | encoded_seq[i + j]
        
        # Calculate position: sum_{j=1}^{k} (i+j) = k*i + k*(k+1)/2
        position = k * i + position_offset
        
        # Update statistics
        counts[idx] += 1
        sum_pos[idx] += position
        sum_sq_pos[idx] += position * position
    
    # Prepare output array
    total_features = 3 * n_kmers
    features = np.zeros(total_features, dtype=np.float64)
    
    # Normalization factor: N - (k - 1) = N - k + 1
    norm_factor = float(N - k + 1)
    
    # Compute and store features
    for idx in range(n_kmers):
        count = counts[idx]
        
        # 1. Frequency
        features[idx] = count
        
        # 2. Average position
        if count > 0:
            avg_pos = sum_pos[idx] / count
            features[n_kmers + idx] = avg_pos
            
            # 3. Center second moment
            avg_sq = sum_sq_pos[idx] / count
            variance = avg_sq - (avg_pos * avg_pos)
            features[2 * n_kmers + idx] = variance / norm_factor if variance > 0 else 0.0
        else:
            features[n_kmers + idx] = 0.0
            features[2 * n_kmers + idx] = 0.0
    
    return features

@jit(nopython=True, parallel=True, fastmath=True)
def _batch_compute_kmer_features_numba(encoded_seqs: list, lengths: np.ndarray,
                                     k: int, position_offset: int, 
                                     n_kmers: int) -> np.ndarray:
    """
    Batch compute k-mer features for multiple sequences.
    This is a standalone function because Numba cannot compile class methods.
    
    Parameters:
    -----------
    encoded_seqs : list of np.ndarray
        List of integer-encoded sequences
    lengths : np.ndarray
        Array of sequence lengths
    k : int
        Size of k-mer
    position_offset : int
        Offset for position calculation: k*(k+1)/2
    n_kmers : int
        Total number of possible k-mers (4^k)
        
    Returns:
    --------
    np.ndarray of shape (n_sequences, 3 * 4^k)
    """
    n_sequences = len(encoded_seqs)
    total_features = 3 * n_kmers
    features_matrix = np.zeros((n_sequences, total_features), dtype=np.float64)
    
    for i in prange(n_sequences):
        features_matrix[i, :] = _compute_kmer_features_numba(
            encoded_seqs[i], lengths[i], k, position_offset, n_kmers
        )
    
    return features_matrix

@jit(nopython=True, parallel=False, cache=True, fastmath=True)
def _compute_kmer_features_optimized(encoded_seq: np.ndarray, N: int, 
                                   k: int, position_offset: int, 
                                   n_kmers: int) -> np.ndarray:
    """
    Optimized computation using pattern matching.
    This is a standalone function because Numba cannot compile class methods.
    """
    # Initialize accumulators
    counts = np.zeros(n_kmers, dtype=np.int32)
    sum_pos = np.zeros(n_kmers, dtype=np.float64)
    sum_sq_pos = np.zeros(n_kmers, dtype=np.float64)
    
    if N < k:
        total_features = 3 * n_kmers
        return np.zeros(total_features, dtype=np.float64)
    
    # Process each k-mer in the sequence
    for i in range(N - k + 1):
        # Calculate k-mer index efficiently
        idx = 0
        for j in range(k):
            idx = (idx << 2) | encoded_seq[i + j]
        
        # Calculate position
        position = k * i + position_offset
        
        # Update statistics
        counts[idx] += 1
        sum_pos[idx] += position
        sum_sq_pos[idx] += position * position
    
    # Compute features
    total_features = 3 * n_kmers
    features = np.zeros(total_features, dtype=np.float64)
    norm_factor = float(N - k + 1)
    
    for idx in range(n_kmers):
        n = counts[idx]
        features[idx] = n
        
        if n > 0:
            u = sum_pos[idx] / n
            features[n_kmers + idx] = u
            
            u2 = sum_sq_pos[idx] / n
            var = u2 - (u * u)
            features[2 * n_kmers + idx] = var / norm_factor if var > 0 else 0.0
        else:
            features[n_kmers + idx] = 0.0
            features[2 * n_kmers + idx] = 0.0
    
    return features

# ============================================================================
# KMerFeatureExtractor class (using the standalone Numba functions)
# ============================================================================

class KMerFeatureExtractor:
    """
    Generalized k-mer feature extractor for DNA sequences.
    
    Extracts three types of features for all possible k-mers:
    1. Frequency (count)
    2. Average position
    3. Center second moment (variance normalized by sequence length)
    
    Total features per sequence: 3 * (4^k)
    """
    
    def __init__(self, k: int = 3):
        """
        Initialize the k-mer feature extractor.
        
        Parameters:
        -----------
        k : int
            Size of k-mer (1 for mononucleotides, 2 for dinucleotides, etc.)
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        
        self.k = k
        self.n_kmers = 4 ** k  # Total number of possible k-mers
        self.total_features = 3 * self.n_kmers
        
        # Pre-compute position offset formula: sum_{j=1}^{k} j = k*(k+1)/2
        self.position_offset = k * (k + 1) // 2
        
        # Pre-compute k-mer index mapping for efficiency
        self._build_kmer_index_mapping()
        
    def _build_kmer_index_mapping(self):
        """Build mapping between k-mer strings and their indices."""
        nucleotides = ['A', 'C', 'G', 'T']
        
        # Generate all possible k-mers
        self.kmer_list = [''.join(comb) for comb in itertools.product(nucleotides, repeat=self.k)]
        
        # Create mapping from k-mer string to index
        self.kmer_to_index = {kmer: idx for idx, kmer in enumerate(self.kmer_list)}
        
        # Create mapping from index to k-mer
        self.index_to_kmer = {idx: kmer for kmer, idx in self.kmer_to_index.items()}
        
        # Create mapping for integer encoding
        self.base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.int_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    
    def encode_sequence(self, sequence: str) -> np.ndarray:
        """
        Encode a DNA sequence as integers.
        
        Parameters:
        -----------
        sequence : str
            DNA sequence
            
        Returns:
        --------
        np.ndarray of encoded integers (A=0, C=1, G=2, T=3)
        """
        seq_upper = sequence.upper()
        encoded = np.zeros(len(seq_upper), dtype=np.int8)
        
        for i, nt in enumerate(seq_upper):
            if nt in self.base_to_int:
                encoded[i] = self.base_to_int[nt]
            else:
                # Handle non-standard nucleotides (map to A)
                encoded[i] = 0
        
        return encoded
    
    def kmer_index(self, encoded_kmer: np.ndarray) -> int:
        """
        Calculate the index (0 to 4^k-1) for a k-mer.
        
        Parameters:
        -----------
        encoded_kmer : np.ndarray
            Array of k integers representing the k-mer
            
        Returns:
        --------
        int : Index of the k-mer
        """
        idx = 0
        for base in encoded_kmer:
            idx = (idx << 2) | base  # Shift left 2 bits and OR with base
        return idx
    
    def compute_features_single(self, sequence: str) -> np.ndarray:
        """
        Compute k-mer features for a single sequence.
        
        Parameters:
        -----------
        sequence : str
            DNA sequence
            
        Returns:
        --------
        np.ndarray of shape (3 * 4^k,) containing all features
        """
        encoded_seq = self.encode_sequence(sequence)
        N = len(encoded_seq)
        
        # Call the standalone Numba function
        return _compute_kmer_features_numba(
            encoded_seq, N, self.k, self.position_offset, self.n_kmers
        )
    
    def compute_features_batch(self, sequences: List[str]) -> np.ndarray:
        """
        Compute k-mer features for a batch of sequences.
        
        Parameters:
        -----------
        sequences : list of str
            List of DNA sequences
            
        Returns:
        --------
        np.ndarray of shape (n_sequences, 3 * 4^k)
        """
        # Encode all sequences
        encoded_seqs = []
        seq_lengths = np.zeros(len(sequences), dtype=np.int32)
        
        for i, seq in enumerate(sequences):
            encoded = self.encode_sequence(seq)
            encoded_seqs.append(encoded)
            seq_lengths[i] = len(encoded)
        
        # Call the standalone Numba batch function
        return _batch_compute_kmer_features_numba(
            encoded_seqs, seq_lengths, self.k, self.position_offset, self.n_kmers
        )
    
    def compute_features_optimized(self, sequence: str) -> np.ndarray:
        """
        Compute k-mer features using the optimized implementation.
        
        Parameters:
        -----------
        sequence : str
            DNA sequence
            
        Returns:
        --------
        np.ndarray of shape (3 * 4^k,) containing all features
        """
        encoded_seq = self.encode_sequence(sequence)
        N = len(encoded_seq)
        
        # Call the optimized standalone Numba function
        return _compute_kmer_features_optimized(
            encoded_seq, N, self.k, self.position_offset, self.n_kmers
        )
    
    def get_feature_names(self) -> List[str]:
        """
        Get descriptive names for all features.
        
        Returns:
        --------
        List of feature names
        """
        feature_types = ['freq', 'avg_pos', 'center_moment']
        feature_names = []
        
        for kmer in self.kmer_list:
            for ft in feature_types:
                feature_names.append(f"{kmer}_{ft}")
        
        return feature_names
    
    def get_kmer_from_index(self, index: int) -> str:
        """
        Convert index to k-mer string.
        
        Parameters:
        -----------
        index : int
            Index of the k-mer (0 to 4^k-1)
            
        Returns:
        --------
        str : k-mer string
        """
        return self.index_to_kmer.get(index, "UNKNOWN")
    
    def get_index_from_kmer(self, kmer: str) -> int:
        """
        Convert k-mer string to index.
        
        Parameters:
        -----------
        kmer : str
            k-mer string
            
        Returns:
        --------
        int : Index of the k-mer
        """
        return self.kmer_to_index.get(kmer.upper(), -1)
    
    def extract_kmer_statistics(self, sequence: str, kmer: str) -> dict:
        """
        Extract detailed statistics for a specific k-mer.
        
        Parameters:
        -----------
        sequence : str
            DNA sequence
        kmer : str
            Specific k-mer to analyze
            
        Returns:
        --------
        dict : Dictionary containing frequency, positions, average, and variance
        """
        if len(kmer) != self.k:
            raise ValueError(f"k-mer length must be {self.k}")
        
        encoded_seq = self.encode_sequence(sequence)
        N = len(encoded_seq)
        
        # Find all occurrences
        positions = []
        encoded_kmer = np.array([self.base_to_int[base] for base in kmer.upper()], dtype=np.int8)
        kmer_idx = self.kmer_index(encoded_kmer)
        
        for i in range(N - self.k + 1):
            current_idx = 0
            for j in range(self.k):
                current_idx = (current_idx << 2) | encoded_seq[i + j]
            
            if current_idx == kmer_idx:
                position = self.k * i + self.position_offset
                positions.append(position)
        
        # Compute statistics
        freq = len(positions)
        
        if freq > 0:
            positions_array = np.array(positions, dtype=np.float64)
            avg_pos = np.mean(positions_array)
            variance = np.var(positions_array)
            norm_factor = N - self.k + 1
            center_moment = variance / norm_factor if variance > 0 else 0.0
        else:
            avg_pos = 0.0
            variance = 0.0
            center_moment = 0.0
        
        return {
            'kmer': kmer,
            'frequency': freq,
            'positions': positions,
            'average_position': avg_pos,
            'variance': variance,
            'center_second_moment': center_moment
        }

# ============================================================================
# Specialized extractors for common k values (for backward compatibility)
# ============================================================================

class MonoNucleotideExtractor(KMerFeatureExtractor):
    """Specialized extractor for k=1 (12 features)."""
    def __init__(self):
        super().__init__(k=1)

class DiNucleotideExtractor(KMerFeatureExtractor):
    """Specialized extractor for k=2 (48 features)."""
    def __init__(self):
        super().__init__(k=2)

class TriNucleotideExtractor(KMerFeatureExtractor):
    """Specialized extractor for k=3 (192 features)."""
    def __init__(self):
        super().__init__(k=3)

class TetraNucleotideExtractor(KMerFeatureExtractor):
    """Specialized extractor for k=4 (768 features)."""
    def __init__(self):
        super().__init__(k=4)