#https://link.springer.com/article/10.1186/s12859-021-04223-3#Sec2

import numpy as np
import scipy.io as sio
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eig
from Bio import SeqIO
from typing import List, Tuple, Dict, Any
import os
from numba import jit, prange, float64, int32, boolean
import numba as nb

@jit(nopython=True, cache=True)
def coordinate() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates points on a circle and transformation vectors.
    
    Returns:
        P: 20x3 array of points
        V: 20x20x3 array of vectors
    """
    n_points = 20
    P = np.zeros((n_points, 3))
    V = np.zeros((n_points, n_points, 3))
    
    # Vectorized computation for P
    angles = np.arange(1, n_points + 1) * 2 * np.pi / n_points
    P[:, 0] = np.cos(angles)
    P[:, 1] = np.sin(angles)
    P[:, 2] = 1
    
    # Vectorized computation for V
    # Reshape P for broadcasting: (20, 1, 3) and (1, 20, 3)
    P_i = P[:, np.newaxis, :]  # Shape: (20, 1, 3)
    P_j = P[np.newaxis, :, :]  # Shape: (1, 20, 3)
    V = P_i + 0.25 * (P_j - P_i)
    
    return P, V


@jit(nopython=True, cache=True)
def compute_sdist_numba(E: np.ndarray, x: int) -> np.ndarray:
    """
    Numba-optimized computation of cumulative distance matrix.
    
    Args:
        E: Distance matrix (x x x)
        x: Dimension of matrix
        
    Returns:
        sdist: Cumulative distance matrix
    """
    sdist = np.zeros((x, x))
    
    for i in range(x):
        for j in range(i, x):
            if j - i == 1:
                sdist[i, j] = E[i, j]
            elif j - i > 1:
                sdist[i, j] = sdist[i, j-1] + E[j-1, j]
    
    return sdist


def ME(W: np.ndarray) -> float:
    """
    Computes the maximum eigenvalue feature from a matrix.
    
    Args:
        W: Input matrix
        
    Returns:
        Maximum eigenvalue normalized by number of rows
    """
    # Remove first row (MATLAB indexing starts at 1)
    W = W[1:, :]
    x = W.shape[0]
    
    # Compute distance matrix
    D = pdist(W)
    E = squareform(D)
    
    # Use numba-optimized function for the nested loops
    sdist = compute_sdist_numba(E, x)
    
    # Make symmetric
    sd = sdist + sdist.T
    
    # Add identity to diagonal
    sdd = sd + np.eye(x)
    
    # Compute normalized matrix
    L = E / sdd
    
    # Compute maximum eigenvalue
    eig_vals = eig(L)
    return np.real(eig_vals[0]) / x


def load_M_mat(filepath: str = 'M.mat') -> List[str]:
    """
    Loads the M.mat file containing 158 strings of 20 characters each.
    
    Args:
        filepath: Path to M.mat file
        
    Returns:
        List of 158 strings
    """
    mat_data = sio.loadmat(filepath)
    # Assuming the variable name is 'M' in the .mat file
    M_array = mat_data['M']
    
    # Convert to list of strings
    M_strings = []
    for i in range(M_array.shape[0]):
        # Extract the string from the array element
        if isinstance(M_array[i][0], np.ndarray):
            # Handle nested array structure
            char_array = M_array[i][0]
            if len(char_array) > 0:
                M_strings.append(''.join([chr(c[0]) if isinstance(c, np.ndarray) else c 
                                         for c in char_array]))
        else:
            # Direct string access
            M_strings.append(str(M_array[i][0]))
    
    return M_strings[:158]  # Ensure we have exactly 158 strings


def GRS(seq: str, P: np.ndarray, V: np.ndarray, M: List[str]) -> List[np.ndarray]:
    """
    Geometric Representation of Sequence.
    
    Args:
        seq: Protein sequence string
        P: Points array from coordinate()
        V: Vectors array from coordinate()
        M: List of 158 strings from M.mat
        
    Returns:
        List of 158 arrays, each of shape (len(seq)+1, 3)
    """
    l_seq = len(seq)
    k = len(M)
    g = []
    
    for j in range(k):
        # Create an array to store all c values
        c_array = np.zeros((l_seq + 1, 3))
        c_array[0] = [0, 0, 0]
        
        # Initialize d
        d = np.zeros(3)
        
        # Reset y for each sequence (previous x)
        y = np.zeros(20, dtype=bool)
        
        for i in range(l_seq):
            # Create boolean array for current amino acid
            x = np.array([aa == seq[i] for aa in M[j]])
            
            if i == 0:  # First position
                if x.any():
                    c_array[i+1] = c_array[i] + P[x][0]
                else:
                    c_array[i+1] = c_array[i]
            else:
                if not x.any():  # Current amino acid not in M[j]
                    d = d * (i - 1) / i if i > 1 else d
                    c_array[i+1] = c_array[i] + np.array([0, 0, 1]) + d
                elif not y.any():  # Previous amino acid not in M[j]
                    d = d * (i - 1) / i if i > 1 else d
                    if x.any():
                        c_array[i+1] = c_array[i] + P[x][0] + d
                    else:
                        c_array[i+1] = c_array[i] + d
                else:  # Both current and previous are in M[j]
                    # Find indices
                    x_idx = np.where(x)[0][0]
                    y_idx = np.where(y)[0][0]
                    d = d * (i - 1) / i + V[y_idx, x_idx] / i if i > 1 else V[y_idx, x_idx]
                    c_array[i+1] = c_array[i] + P[x_idx] + d
            
            # Update y for next iteration
            y = x.copy()
        
        g.append(c_array)
    
    return g


# Create a specialized function signature for numba
@jit(nopython=True, cache=True)
def count_aac_dpc_numba(seq_indices: np.ndarray, len_seq: int, len_a: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-optimized counting of amino acid and dipeptide compositions.
    
    Args:
        seq_indices: Array of amino acid indices (0-19 for valid, -1 for invalid)
        len_seq: Length of sequence
        len_a: Number of amino acids (20)
        
    Returns:
        AAC: Amino Acid Composition vector
        DPC: Dipeptide Composition matrix
    """
    # Initialize AAC and DPC
    AAC = np.zeros(len_a, dtype=np.float64)
    DPC = np.zeros((len_a, len_a), dtype=np.float64)
    
    # Count amino acids
    for i in range(len_seq):
        idx = seq_indices[i]
        if idx >= 0:  # Valid amino acid
            AAC[idx] += 1.0
    
    # Normalize AAC by sequence length
    if len_seq > 0:
        for i in range(len_a):
            AAC[i] /= len_seq
    
    # Count dipeptides
    if len_seq > 1:
        for i in range(len_seq - 1):
            idx1 = seq_indices[i]
            idx2 = seq_indices[i + 1]
            
            if idx1 >= 0 and idx2 >= 0:  # Both valid
                DPC[idx1, idx2] += 1.0
        
        # Normalize DPC by (len_seq - 1)
        norm_factor = 1.0 / (len_seq - 1)
        for i in range(len_a):
            for j in range(len_a):
                DPC[i, j] *= norm_factor
    
    return AAC, DPC


def SAD(seq: str, char_set: str = 'ARNDCQEGHILKMFPSTWYV') -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Amino Acid Composition (AAC) and Dipeptide Composition (DPC).
    
    Args:
        seq: Protein sequence string
        char_set: String of allowed amino acids
        
    Returns:
        AAC: Amino Acid Composition vector (20,)
        DPC: Dipeptide Composition matrix (20, 20)
    """
    len_seq = len(seq)
    len_a = len(char_set)
    
    # Create character to index mapping
    char_to_idx = {char: i for i, char in enumerate(char_set)}
    
    # Convert sequence to indices (using -1 for invalid characters)
    seq_indices = np.full(len_seq, -1, dtype=np.int32)
    for i, aa in enumerate(seq):
        if aa in char_to_idx:
            seq_indices[i] = char_to_idx[aa]
    
    # Use numba-optimized function for counting
    AAC, DPC = count_aac_dpc_numba(seq_indices, len_seq, len_a)
    
    return AAC, DPC


def FEGS(data: str) -> np.ndarray:
    """
    Main function to compute features for protein sequences.
    
    Args:
        data: Number or identifier for the FASTA file
        
    Returns:
        FV: Feature matrix of shape (n_sequences, n_features)
    """
    # Load coordinate system
    P, V = coordinate()
    
    # Load M matrix
    M = load_M_mat('M.mat')
    
    # Read FASTA file
    fasta_file = f"{data}.fasta"
    
    # Check if file exists
    if not os.path.exists(fasta_file):
        raise FileNotFoundError(f"FASTA file {fasta_file} not found")
    
    # Read sequences using Biopython
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append([str(record.seq)])
    
    l = len(sequences)
    
    # Initialize feature arrays
    EL = np.zeros((l, 158))  # Eigenvalue features
    FA = np.zeros((l, 20))   # AAC features
    FD = np.zeros((l, 400))  # DPC features (20x20 flattened)
    
    # Process each sequence
    for i, seq in enumerate(sequences):
        # Compute GRS features
        g_p = GRS(seq, P, V, M)
        
        # Compute ME for each of 158 representations
        for u in range(158):
            EL[i, u] = ME(g_p[u])
        
        # Compute AAC and DPC
        AAC, DPC = SAD(seq)
        FA[i, :] = AAC
        FD[i, :] = DPC.flatten()
    
    # Concatenate all features
    FV = np.hstack([EL, FA, FD])
    
    return FV


if __name__ == "__main__":
    features = FEGS('data')
    print(features)