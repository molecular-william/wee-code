#https://github.com/SAkbari93/PCV-method
#https://www.nature.com/articles/s41598-022-15266-8#Abs1
import numpy as np
import numba as nb
from scipy.io import loadmat
from Bio import SeqIO
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
class Config:
    """Configuration parameters matching the MATLAB code"""
    Lm = 50  # Block length
    Lshift = 5  # Shift range
    AMINO_ACIDS = list('ARNDCQEGHILKMFPSTWYV')
    UNKNOWN_AA = 'X'
    
# ==================== Numba-optimized Functions ====================
@nb.njit(nogil=True, cache=True)
def normalize_rows(data: np.ndarray) -> np.ndarray:
    """Normalize data matrix (mean=0, std=1) along rows - Numba compatible"""
    n_rows, n_cols = data.shape
    norm_data = np.zeros_like(data)
    
    for i in range(n_rows):
        # Compute mean
        mean_val = 0.0
        count = 0
        for j in range(n_cols):
            val = data[i, j]
            if not np.isnan(val):
                mean_val += val
                count += 1
        if count > 0:
            mean_val /= count
        
        # Compute standard deviation
        std_val = 0.0
        for j in range(n_cols):
            val = data[i, j]
            if not np.isnan(val):
                diff = val - mean_val
                std_val += diff * diff
        if count > 1:
            std_val = np.sqrt(std_val / (count - 1))
        else:
            std_val = 1.0
        
        # Avoid division by zero
        if std_val == 0:
            std_val = 1.0
        
        # Normalize
        for j in range(n_cols):
            val = data[i, j]
            if np.isnan(val):
                norm_data[i, j] = 0.0
            else:
                norm_data[i, j] = (val - mean_val) / std_val
    
    return norm_data

@nb.njit(nogil=True, cache=True)
def compute_moment_numba(seq: str, pattern_groups_flat: np.ndarray, 
                        pattern_group_sizes: np.ndarray) -> np.ndarray:
    """
    Compute moment features for a sequence - Numba compatible
    
    Parameters:
    -----------
    seq : str
        Amino acid sequence
    pattern_groups_flat : np.ndarray
        Flattened array of all pattern characters
    pattern_group_sizes : np.ndarray
        Array containing the size of each pattern group
    
    Returns:
    --------
    np.ndarray: Moment features for the sequence
    """
    n = len(seq)
    ngrp = len(pattern_group_sizes)
    
    # Initialize arrays
    D = np.zeros(ngrp, dtype=np.float64)
    N = np.zeros(ngrp, dtype=np.float64)
    Mu = np.zeros(ngrp, dtype=np.float64)
    
    # Pre-compute starting indices for each pattern group
    start_indices = np.zeros(ngrp + 1, dtype=np.int32)
    for i in range(ngrp):
        start_indices[i + 1] = start_indices[i] + pattern_group_sizes[i]
    
    # First pass: count occurrences and compute weighted positions
    for i in range(n):
        aa = seq[i]
        pos = i + 1  # 1-based indexing like MATLAB
        
        # Check which pattern group contains this amino acid
        for group_idx in range(ngrp):
            start_idx = start_indices[group_idx]
            end_idx = start_indices[group_idx + 1]
            
            found = False
            # Check all characters in this pattern group
            for j in range(start_idx, end_idx):
                if pattern_groups_flat[j] == aa:
                    N[group_idx] += 1
                    Mu[group_idx] += pos
                    found = True
                    break
            
            # Break if found (each AA belongs to only one group)
            if found:
                break
    
    # Compute means (avoid division by zero)
    for r in range(ngrp):
        if N[r] > 0:
            Mu[r] = Mu[r] / N[r]
    
    # Second pass: compute variance
    for i in range(n):
        aa = seq[i]
        pos = i + 1  # 1-based indexing
        
        # Check which pattern group contains this amino acid
        for group_idx in range(ngrp):
            start_idx = start_indices[group_idx]
            end_idx = start_indices[group_idx + 1]
            
            found = False
            # Check all characters in this pattern group
            for j in range(start_idx, end_idx):
                if pattern_groups_flat[j] == aa:
                    diff = pos - Mu[group_idx]
                    D[group_idx] += diff * diff
                    found = True
                    break
            
            # Break if found
            if found:
                break
    
    # Normalize and handle edge cases
    for r in range(ngrp):
        if N[r] > 0 and n > 0:
            D[r] = D[r] / (N[r] * n)
        else:
            D[r] = 0.0
    
    return D

@nb.njit(nogil=True, cache=True, parallel=True)
def compute_distances_numba(F_data: np.ndarray) -> np.ndarray:
    """
    Compute distance matrix using vectorized operations
    F_data shape: (num_seq, num_shifts, num_blocks, feature_dim)
    """
    num_seq, num_shifts, num_blocks, feature_dim = F_data.shape
    out3 = np.zeros((num_seq, num_seq))
    
    for u in nb.prange(num_seq):
        for v in range(num_seq):
            total_dist = 0.0
            
            for p in range(num_blocks):
                min_dist = np.inf
                
                for i in range(num_shifts):
                    G1 = F_data[u, i, p, :]
                    
                    for j in range(num_shifts):
                        G2 = F_data[v, j, p, :]
                        
                        # Max absolute difference
                        max_diff = 0.0
                        for k in range(feature_dim):
                            diff = abs(G1[k] - G2[k])
                            if diff > max_diff:
                                max_diff = diff
                        
                        if max_diff < min_dist:
                            min_dist = max_diff
                
                total_dist += min_dist
            
            out3[u, v] = total_dist
    
    return out3

# ==================== Main Processing Class ====================
class ProteinVectorizer:
    """Main class for protein sequence vectorization"""
    
    def __init__(self, config: Config):
        self.config = config
        self.P = {}  # Normalized PCH data
        self.pattern_groups = []  # Pattern groups for moments
        
    def process_pch_data(self, avg_data: np.ndarray) -> None:
        """Process and normalize PCH data"""
        # Handle NaN values
        DD = np.where(np.isnan(avg_data), 0, avg_data)
        
        # Normalize using Numba-compatible function
        norm_data = normalize_rows(DD)
        
        # Create P dictionary for amino acids
        for idx, aa in enumerate(self.config.AMINO_ACIDS):
            self.P[aa] = norm_data[:, idx]
        
        # Add unknown amino acid
        self.P[self.config.UNKNOWN_AA] = np.zeros(norm_data.shape[0])
    
    def pad_sequences(self, sequences: list) -> tuple:
        """Pad sequences to maximum length with 'X'"""
        # Find max length
        Lmax = max(len(seq) for seq in sequences)
        Le = self.config.Lm * (Lmax // self.config.Lm + 1)
        
        # Pad sequences
        padded_seqs = []
        for seq in sequences:
            padded = seq + self.config.UNKNOWN_AA * (Le - len(seq))
            padded_seqs.append(padded)
        
        return padded_seqs, Le // self.config.Lm
    
    def create_blocks(self, seq: str) -> np.ndarray:
        """
        Create shifted blocks from a sequence
        Returns: (num_shifts, num_blocks, block_len) array
        """
        seq_len = len(seq)
        num_shifts = 2 * self.config.Lshift + 1
        num_blocks = seq_len // self.config.Lm
        
        # Initialize array for characters
        blocks = np.empty((num_shifts, num_blocks, self.config.Lm), dtype='U1')
        
        for k in range(-self.config.Lshift, self.config.Lshift + 1):
            shift_idx = k + self.config.Lshift
            
            # Apply circular shift with padding
            if k > 0:
                # Shift right, pad left
                shifted = seq[k:] + self.config.UNKNOWN_AA * k
            elif k < 0:
                # Shift left, pad right
                shifted = self.config.UNKNOWN_AA * (-k) + seq[:k]
            else:
                shifted = seq
            
            # Ensure length matches
            if len(shifted) < seq_len:
                shifted += self.config.UNKNOWN_AA * (seq_len - len(shifted))
            elif len(shifted) > seq_len:
                shifted = shifted[:seq_len]
            
            # Reshape into blocks
            for i in range(num_blocks):
                start_idx = i * self.config.Lm
                end_idx = start_idx + self.config.Lm
                block = shifted[start_idx:end_idx]
                blocks[shift_idx, i, :] = list(block)
        
        return blocks
    
    def compute_pch_features(self, blocks: np.ndarray) -> np.ndarray:
        """Compute PCH features from blocks"""
        num_shifts, num_blocks, block_len = blocks.shape
        feature_dim = len(self.P['A'])  # Dimension of PCH features
        
        # Initialize feature array
        features = np.zeros((num_shifts, num_blocks, feature_dim))
        
        for shift_idx in range(num_shifts):
            for block_idx in range(num_blocks):
                block = blocks[shift_idx, block_idx, :]
                feature_sum = np.zeros(feature_dim)
                
                for aa in block:
                    if aa in self.P:
                        feature_sum += self.P[aa]
                
                features[shift_idx, block_idx, :] = feature_sum
        
        return features
    
    def compute_moment_features(self, blocks: np.ndarray) -> np.ndarray:
        """Compute moment features from blocks"""
        if not self.pattern_groups:
            raise ValueError("Pattern groups not loaded. Call load_patterns() first.")
        
        num_shifts, num_blocks, block_len = blocks.shape
        ngrp = len(self.pattern_groups)
        
        # Initialize feature array
        features = np.zeros((num_shifts, num_blocks, ngrp))
        
        # Flatten pattern groups for Numba compatibility
        pattern_groups_flat = []
        pattern_group_sizes = np.zeros(ngrp, dtype=np.int32)
        
        for i, group in enumerate(self.pattern_groups):
            pattern_group_sizes[i] = len(group)
            pattern_groups_flat.extend(group)
        
        pattern_groups_flat = np.array(pattern_groups_flat, dtype='U1')
        
        for shift_idx in range(num_shifts):
            for block_idx in range(num_blocks):
                # Convert block to string
                block_str = ''.join(blocks[shift_idx, block_idx, :])
                # Compute moments
                moments = compute_moment_numba(block_str, pattern_groups_flat, pattern_group_sizes)
                features[shift_idx, block_idx, :] = moments
        
        # Normalize moment features
        for shift_idx in range(num_shifts):
            data = features[shift_idx]
            if data.size > 0:
                # Normalize per block
                features[shift_idx] = normalize_rows(data)
        
        return features
    
    def merge_features(self, pch_features: np.ndarray, moment_features: np.ndarray) -> np.ndarray:
        """Merge PCH and moment features"""
        return np.concatenate([pch_features, moment_features], axis=-1)
    
    def compute_distance_matrix(self, all_features: np.ndarray) -> np.ndarray:
        """Compute distance matrix for all sequences"""
        return compute_distances_numba(all_features)
    
    def save_mega_format(self, distance_matrix: np.ndarray, ids: list, 
                        output_file: str) -> None:
        """Save distance matrix in MEGA format"""
        num_seq = len(ids)
        
        with open(output_file, 'w') as f:
            f.write('#mega\n')
            f.write('!Title: TEST;\n')
            f.write(f'!Format DataType=Distance DataFormat=LowerLeft NTaxa={num_seq};\n\n')
            
            # Write sequence IDs
            for k, seq_id in enumerate(ids, 1):
                f.write(f'[{k}] #{seq_id}\n')
            
            f.write('\n')
            
            # Write distance matrix (lower triangular)
            for j in range(1, num_seq):
                f.write(f'[{j+1}]   ')
                for k in range(j):
                    f.write(f' {distance_matrix[j, k]:8.6f}')
                f.write('\n')
            f.write('\n')

# ==================== Data Loading Functions ====================
def load_fasta(filename: str) -> tuple:
    """Load FASTA file and return IDs and sequences"""
    sequences = []
    ids = []
    
    for record in SeqIO.parse(filename, "fasta"):
        ids.append(record.id)
        sequences.append(str(record.seq))
    
    return ids, sequences

def load_mat_data(mat_file: str, data_key: str = 'Avgdata'):
    """Load MATLAB .mat file data"""
    data = loadmat(mat_file)
    return data[data_key]

def load_patterns(mat_file: str, pattern_key: str = 'PatternG'):
    """Load pattern groups from MATLAB file"""
    data = loadmat(mat_file)
    patterns = data[pattern_key]
    
    # Convert to list of lists
    pattern_list = []
    for i in range(patterns.shape[0]):
        # Handle different MATLAB array structures
        cell_array = patterns[i][0]
        pattern_group = []
        for item in cell_array:
            # Extract the string from MATLAB cell
            if isinstance(item, np.ndarray):
                for elem in item:
                    if isinstance(elem, str):
                        pattern_group.append(elem.strip())
                    else:
                        # Try to convert to string
                        pattern_group.append(str(elem).strip())
            elif isinstance(item, str):
                pattern_group.append(item.strip())
        pattern_list.append(pattern_group)
    
    return pattern_list

# ==================== Main Execution ====================
def main():
    """Main execution function"""
    # Configuration
    config = Config()
    name = 'Betaglobin-50'
    
    # Initialize vectorizer
    vectorizer = ProteinVectorizer(config)
    
    try:
        # 1. Read FASTA file
        print("Reading FASTA file...")
        fasta_file = f"{name}.fasta"
        ids, sequences = load_fasta(fasta_file)
        print(f"Loaded {len(ids)} sequences")
        
        # 2. Load and process PCH data
        print("Loading PCH data...")
        avg_data = load_mat_data('Avgdata.mat')
        print(f"PCH data shape: {avg_data.shape}")
        vectorizer.process_pch_data(avg_data)
        
        # 3. Load pattern groups
        print("Loading pattern groups...")
        pattern_groups = load_patterns('PatternG.mat')
        vectorizer.pattern_groups = pattern_groups
        print(f"Loaded {len(pattern_groups)} pattern groups")
        
        # 4. Pad sequences and create blocks
        print("Processing sequences...")
        padded_seqs, num_blocks = vectorizer.pad_sequences(sequences)
        print(f"Created {num_blocks} blocks per sequence")
        
        # 5. Process all sequences
        all_features = []
        
        for seq_idx, seq in enumerate(padded_seqs):
            if (seq_idx + 1) % 10 == 0 or seq_idx == 0:
                print(f"Processing sequence {seq_idx + 1}/{len(padded_seqs)}...")
            
            # Create blocks
            blocks = vectorizer.create_blocks(seq)
            
            # Compute PCH features
            pch_features = vectorizer.compute_pch_features(blocks)
            
            # Compute moment features
            moment_features = vectorizer.compute_moment_features(blocks)
            
            # Merge features
            merged_features = vectorizer.merge_features(pch_features, moment_features)
            
            all_features.append(merged_features)
        
        # Convert to numpy array
        all_features = np.array(all_features)
        print(f"Feature array shape: {all_features.shape}")
        
        # 6. Compute distance matrix
        print("Computing distance matrix...")
        distance_matrix = vectorizer.compute_distance_matrix(all_features)
        
        print(f"Distance matrix shape: {distance_matrix.shape}")
        print("Distance matrix:")
        print(distance_matrix)
        
        # 7. Save in MEGA format
        output_file = f"PCV_{name}.meg"
        print(f"Saving results to {output_file}...")
        vectorizer.save_mega_format(distance_matrix, ids, output_file)
        
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Running full analysis...")
    print("="*50)
    main()