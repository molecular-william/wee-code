"""
@Env: /anaconda3/python3.11
@Time: 2025/4/16-10:04
@Auth: karlieswift
@File: eev_optimized.py
@Desc: Optimized version of EnergyEntropy with numba compilation and vectorization
"""

import numpy as np
import itertools
from numba import jit, prange, int32, float64, typed, types
from collections import defaultdict
import time
from functools import lru_cache


class EnergyEntropyOptimized:
    def __init__(self, data_type="DNA", energy_values=2, mutual_information_energy=2, use_numba=True):
        """
        Optimized EnergyEntropy with vectorization and numba compilation
        
        data_type (str): The data type can be "protein", "DNA" or "RNA"
        energy_values (int): The incoming energy value: 1,2,3,4.....
        mutual_information_energy (int): The energy of mutual information : 2,3,4.....
        use_numba (bool): Whether to use numba JIT compilation
        """
        self.data_type = data_type
        self.energy_values = energy_values
        self.mutual_information_energy = mutual_information_energy
        self.use_numba = use_numba
        
        # Get elements and create mapping
        self.data_type_elements = self._get_data_type(data_type=data_type)
        self.char_to_int = {char: i for i, char in enumerate(self.data_type_elements)}
        self.int_to_char = {i: char for i, char in enumerate(self.data_type_elements)}
        self.n_elements = len(self.data_type_elements)
        
        # Precompute combinations for faster access
        self._precompute_combinations()
        
        # Compile numba functions if needed
        if self.use_numba:
            self._compile_numba_functions()
    
    def _get_data_type(self, data_type):
        data_type_dict = {
            "dna": 'ACGT',
            "protein": 'ACDEFGHIKLMNPQRSTVWY',
            "rna": 'ACGU',
        }
        return data_type_dict.get(data_type.lower())
    
    def _precompute_combinations(self):
        """Precompute combinations for faster access"""
        self.combinations_cache = {}
        
        # Precompute all combinations for k=1 to energy_values-1
        for k in range(1, self.energy_values):
            self.combinations_cache[k] = list(itertools.combinations(range(self.n_elements), k))
        
        # Precompute pairs
        self.pairs = list(itertools.product(range(self.n_elements), repeat=2))
        self.pair_indices = {f"{i}{j}": idx for idx, (i, j) in enumerate(self.pairs)}
        
        # Precompute mutual information combinations
        self.mutual_combinations = []
        for r in range(2, self.mutual_information_energy + 1):
            combs = list(itertools.combinations(range(self.n_elements), r))
            self.mutual_combinations.extend([''.join(sorted(str(x) for x in comb)) for comb in combs])
    
    def _compile_numba_functions(self):
        """Compile numba functions on initialization"""
        # Test compilation with dummy data
        dummy_seq = np.array([0, 1, 2, 3], dtype=np.int32)
        dummy_len = len(dummy_seq)
        
        # Compile the functions
        self._numba_count_occurrences(dummy_seq, dummy_len, 4)
        self._numba_count_pairs(dummy_seq, dummy_len, 4)
        self._numba_compute_position_info(dummy_seq, dummy_len, 4)
    
    def seq_to_int_array(self, seq):
        """Convert sequence string to integer array"""
        return np.array([self.char_to_int[char] for char in seq], dtype=np.int32)
    
    def seq2vector(self, seq):
        """Main method to convert sequence to vector - optimized version"""
        # Convert sequence to integer array for faster processing
        seq_int = self.seq_to_int_array(seq)
        seq_len = len(seq_int)
        
        # Use optimized counting functions
        if self.use_numba and seq_len > 10:  # Use numba for longer sequences
            number_X = self._numba_count_occurrences(seq_int, seq_len, self.n_elements)
            pair_counts = self._numba_count_pairs(seq_int, seq_len, self.n_elements)
        else:
            number_X = np.bincount(seq_int, minlength=self.n_elements)
            pair_counts = self._count_pairs_python(seq_int, seq_len)
        
        # Calculate probabilities and entropy E1
        p_X = number_X / seq_len
        H_X = np.zeros(self.n_elements)
        mask = p_X > 0
        H_X[mask] = -p_X[mask] * np.log2(p_X[mask])
        
        # Calculate E1 using vectorized approach
        E1 = self._compute_combinations_vectorized(H_X, number_X, self.energy_values)
        
        # Calculate H_p_point_AGCT for E2
        H_p_point = self._compute_H_p_point_vectorized(pair_counts, seq_len)
        E2 = self._compute_combinations_vectorized(H_p_point, number_X, self.energy_values)
        
        # Calculate position information for E3
        H_relative_position = self._compute_position_entropy(seq_int, seq_len)
        E3 = self._compute_combinations_vectorized(H_relative_position, number_X, self.energy_values)
        
        # Calculate mutual information for E4
        E4 = self._get_mutual_information_vectorized(seq_int, seq_len)
        
        # Combine all vectors
        all_vector = np.concatenate([E1, E2, E3, E4])
        
        return all_vector.tolist()
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _numba_count_occurrences(seq_int, seq_len, n_elements):
        """Numba-optimized count of element occurrences"""
        counts = np.zeros(n_elements, dtype=np.int32)
        for i in prange(seq_len):
            counts[seq_int[i]] += 1
        return counts
    
    @staticmethod
    @jit(nopython=True)
    def _numba_count_pairs(seq_int, seq_len, n_elements):
        """Numba-optimized count of pairs"""
        pair_counts = np.zeros((n_elements, n_elements), dtype=np.int32)
        for i in range(seq_len - 1):
            a = seq_int[i]
            b = seq_int[i + 1]
            pair_counts[a, b] += 1
        return pair_counts
    
    @staticmethod
    @jit(nopython=True)
    def _numba_compute_position_info(seq_int, seq_len, n_elements):
        """Numba-optimized position information calculation"""
        position_sums = np.zeros(n_elements, dtype=np.int32)
        for i in range(seq_len):
            elem = seq_int[i]
            position_sums[elem] += (i + 1)
        return position_sums
    
    def _count_pairs_python(self, seq_int, seq_len):
        """Python fallback for pair counting"""
        pair_counts = np.zeros((self.n_elements, self.n_elements), dtype=np.int32)
        for i in range(seq_len - 1):
            a = seq_int[i]
            b = seq_int[i + 1]
            pair_counts[a, b] += 1
        return pair_counts
    
    def _compute_H_p_point_vectorized(self, pair_counts, seq_len):
        """Vectorized computation of H_p_point"""
        # Normalize pair counts
        total_pairs = np.sum(pair_counts)
        if total_pairs == 0:
            return np.zeros(self.n_elements)
        
        # Calculate entropy for each element as second element in pair
        H_p_point = np.zeros(self.n_elements)
        for e in range(self.n_elements):
            # Get pairs where second element is e
            col_pairs = pair_counts[:, e]
            col_sum = np.sum(col_pairs)
            if col_sum > 0:
                # Normalize
                p_col = col_pairs / col_sum
                # Calculate entropy with small epsilon to avoid log(0)
                mask = p_col > 0
                if np.any(mask):
                    H_p_point[e] = -np.sum(p_col[mask] * np.log2(p_col[mask] + 1e-10))
        
        return H_p_point
    
    def _compute_position_entropy(self, seq_int, seq_len):
        """Optimized computation of position entropy"""
        if self.use_numba:
            position_sums = self._numba_compute_position_info(seq_int, seq_len, self.n_elements)
        else:
            position_sums = np.zeros(self.n_elements, dtype=np.int32)
            for i, elem in enumerate(seq_int):
                position_sums[elem] += (i + 1)
        
        # Calculate all_position
        all_position = (seq_len + 1) * seq_len * 0.5
        
        # Calculate relative position and entropy
        relative_position = all_position - position_sums
        relative_position_p = relative_position / all_position
        
        H_relative_position = np.zeros(self.n_elements)
        mask = relative_position_p > 0
        H_relative_position[mask] = -relative_position_p[mask] * np.log2(relative_position_p[mask])
        
        return H_relative_position
    
    def _compute_combinations_vectorized(self, I_XY, number_XY, k_max):
        """Vectorized computation of combinations"""
        results = []
        
        for k in range(1, k_max):
            # Get precomputed combinations for this k
            combs = self.combinations_cache.get(k, [])
            if not combs:
                continue
            
            # Calculate for all combinations at once using vectorization
            for comb in combs:
                # Convert tuple to list for indexing
                indices = list(comb)
                product_I_XY = np.sum(I_XY[indices])
                product_number_XY = np.sum(number_XY[indices])
                results.append(product_I_XY * product_number_XY)
        
        return np.array(results)
    
    def _get_mutual_information_vectorized(self, seq_int, seq_len):
        """Optimized mutual information calculation"""
        r = self.mutual_information_energy
        if r < 2:
            return np.array([])
        
        # Calculate individual probabilities
        counts = np.bincount(seq_int, minlength=self.n_elements)
        p_i = counts / seq_len
        
        # Get all r-length combinations
        comb_indices = list(itertools.combinations(range(self.n_elements), r))
        
        # Pre-calculate combination probabilities
        comb_probs = []
        comb_strings = []
        
        for indices in comb_indices:
            # Calculate p(x,y,z,...) = p(x)*p(y)*p(z)*...
            prob = 1.0
            for idx in indices:
                prob *= p_i[idx]
            comb_probs.append(prob)
            comb_strings.append(''.join(sorted(str(idx) for idx in indices)))
        
        # Count occurrences of each combination
        comb_counts = defaultdict(int)
        for i in range(seq_len - (r - 1)):
            # Get r consecutive elements and sort them
            window = seq_int[i:i + r]
            sorted_window = np.sort(window)
            # Convert to string key
            key = ''.join(str(x) for x in sorted_window)
            comb_counts[key] += 1
        
        # Calculate mutual information
        energy_info = []
        for comb_str, p_comb in zip(comb_strings, comb_probs):
            count = comb_counts.get(comb_str, 0)
            if count == 0 or p_comb == 0:
                energy_info.append(0.0)
            else:
                p_observed = count / seq_len
                energy_info.append(np.log2(p_observed / p_comb) * p_observed * count)
        
        return np.array(energy_info)
    
    def batch_process(self, sequences):
        """Process multiple sequences in batch"""
        results = []
        for seq in sequences:
            results.append(self.seq2vector(seq))
        return results


# Additional optimized functions that can be used outside the class
@jit(nopython=True, parallel=True)
def fast_pair_counts(seq_int, n_elements):
    """Standalone fast pair counting function"""
    seq_len = len(seq_int)
    pair_counts = np.zeros((n_elements, n_elements), dtype=np.int32)
    
    for i in prange(seq_len - 1):
        a = seq_int[i]
        b = seq_int[i + 1]
        pair_counts[a, b] += 1
    
    return pair_counts


@jit(nopython=True)
def fast_entropy_calculation(probabilities):
    """Fast entropy calculation with numpy"""
    mask = probabilities > 0
    if not np.any(mask):
        return 0.0
    return -np.sum(probabilities[mask] * np.log2(probabilities[mask]))


if __name__ == '__main__':
    # Test sequences
    test_sequences = [
        "GGTCAATTTAAGAGGAAGTAAAAGTCGTAACAAGGTTTCCGTAGGTGAACCTGCGGAAGGATCATTATCGAATCCGA"*1000,
        "AAGATTCGGGCCTTCGGGTCCACCCGTTCCGCAGCTGTGCGCTCTTTGGGCTGCACGCTGTGTGATACACAACCCTCACACCTGTGAACGTATCGGGGGCGCGTAAGCGCTTCTGCTCAAAACATTTAACTACTTATGTTCAGAATGTAAAAAACTATAACAAATAACAACTTTCAACAACGGATCTCTTGGCTCTCGCATCGATGAAGAACGCAGCGAAATGCGATAAGTAATGTGAATTGCAGAATTCAGTGAATCATCGAATCTTTGAACGCACCTTGCGCTCTGTGGTATTCCGCAGAGCATGCCTGTTTGAGTGTCACGTAAACCATCGCCCTTGGGATTTCGATCTCTATGAGGTGGACTTGGACTGTGCCGTAGTCGGCTCGTCTTGAAATGAATTAGCTTGCGCTCTTTAGAGTGTCCGGCACCGGTGTGATAATTATCTGCGCCAACGCCTATGGCCTCTTCTTGCGGTGCTGCTTACAGTAGTCCGAAAGGACAGATCTACTTTAAAGCTTTGGCCTCA",
        "GAGTAAGATC"*10
    ]
    
    # Test with and without numba
    print("Testing optimized EnergyEntropy...")
    
    for use_numba in [True, False]:
        print(f"\n{'='*50}")
        print(f"Testing with use_numba={use_numba}")
        print(f"{'='*50}")
        
        eev = EnergyEntropyOptimized(use_numba=use_numba)
        
        total_time = 0
        for i, seq in enumerate(test_sequences):
            print(f"\nSequence {i+1} (length: {len(seq)}):")
            
            start_time = time.time()
            vector = eev.seq2vector(seq)
            elapsed = time.time() - start_time
            
            print(f"  Time: {elapsed:.4f} seconds")
            print(f"  Vector length: {len(vector)}")
            print(f"  First 10 values: {vector[:10]}")
            
            total_time += elapsed
        
        print(f"\nTotal processing time: {total_time:.4f} seconds")
    
    # Compare with original implementation
    print(f"\n{'='*50}")
    print("Performance comparison with original implementation")
    print(f"{'='*50}")
    
    # Original implementation (simplified for comparison)
    class EnergyEntropyOriginal:
        def __init__(self, data_type="DNA", energy_values=2, mutual_information_energy=2):
            self.data_type = data_type
            self.energy_values = energy_values
            self.mutual_information_energy = mutual_information_energy
            self.data_type_elements = self._get_data_type(data_type=data_type)
        
        def _get_data_type(self, data_type):
            data_type_dict = {"dna": 'ACGT', "protein": 'ACDEFGHIKLMNPQRSTVWY', "rna": 'ACGU'}
            return data_type_dict.get(data_type.lower())
        
        def seq2vector(self, seq):
            # Simplified original implementation for comparison
            seq_len = len(seq)
            number_X = {e: seq.count(e) for e in self.data_type_elements}
            p_X = {e: number_X[e] / seq_len for e in self.data_type_elements}
            
            H_X = {k: 0 for k, v in p_X.items()}
            for k, v in p_X.items():
                if v != 0:
                    H_X[k] = -v * np.log2(v)
            
            # Simple combination calculation
            keys = list(H_X.keys())
            results = []
            for k in range(1, self.energy_values):
                combs = list(itertools.combinations(keys, k))
                for comb in combs:
                    product_I_XY = sum(H_X[key] for key in comb)
                    product_number_XY = sum(number_X[key] for key in comb)
                    results.append(product_I_XY * product_number_XY)
            
            return [float(v) for v in results]
    
    # Run comparison
    test_seq = test_sequences[0]
    
    original = EnergyEntropyOriginal()
    optimized = EnergyEntropyOptimized(use_numba=True)
    
    start = time.time()
    orig_vector = original.seq2vector(test_seq)
    orig_time = time.time() - start
    
    start = time.time()
    opt_vector = optimized.seq2vector(test_seq)
    opt_time = time.time() - start
    
    print(f"\nOriginal implementation:")
    print(f"  Time: {orig_time:.4f} seconds")
    print(f"  Vector length: {len(orig_vector)}")
    
    print(f"\nOptimized implementation:")
    print(f"  Time: {opt_time:.4f} seconds")
    print(f"  Vector length: {len(opt_vector)}")
    
    print(f"\nSpeedup: {orig_time/opt_time:.2f}x faster")