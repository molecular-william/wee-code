#https://github.com/Gmcen20/ANV250
#https://www.mdpi.com/2073-4425/13/10/1744
import numpy as np
import pandas as pd
from Bio import SeqIO
from numba import jit

s1 = ['AGYLLGKINLKALAALAKKILTYADFIASGRTGRRNAI']
n1 = len(s1)

def seq2mats(seq):
    n_counts = np.zeros(20, dtype=int)
    t = np.zeros((20, N))
    miu = np.zeros((20, N))
    # Count amino acids and record positions
    amino_acids = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
        'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
        'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
        'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
    }
    
    for i, aa in enumerate(seq):
        if aa in amino_acids:
            idx = amino_acids[aa]
            n_counts[idx] += 1
            t[idx, n_counts[idx]-1] = i + 1  # +1 for 1-based indexing like MATLAB
            miu[idx, i] += 1
        else:
            print(f"Skipping sequence {m+1} due to non-standard amino acids")
            return
        
    return n_counts, t, miu

@jit(nopython=True)
def anv_process(n_counts, t, miu, N, n1, feature1, t0, theta, sigma, D, kesai, cov):
    # Calculate miu matrix
    for i in range(20):
        if n_counts[i] > 0:
            if t[i, 0] - 1 > 0:
                # Set first positions to 0
                for j in range(int(t[i, 0]) - 1):
                    miu[i, j] = 0
                
                # Fill intermediate positions
                for j in range(1, n_counts[i]):
                    start = int(t[i, j-1]) - 1  # Convert to 0-based
                    end = int(t[i, j]) - 1
                    for k in range(start, end):
                        miu[i, k] = j
                
                # Fill positions after last occurrence
                for k in range(int(t[i, n_counts[i]-1]) - 1, N):
                    miu[i, k] = n_counts[i]
            else:  # t[i, 0] - 1 == 0
                for j in range(1, n_counts[i]):
                    start = int(t[i, j-1]) - 1
                    end = int(t[i, j]) - 1
                    for k in range(start, end):
                        miu[i, k] = j
                
                for k in range(int(t[i, n_counts[i]-1]) - 1, N):
                    miu[i, k] = n_counts[i]
    
    # Calculate theta
    for i in range(20):
        theta[i] = np.sum(miu[i, :]) / N
    
    # Calculate sigma
    for i in range(20):
        sigma[i] = np.sum(miu[i, :])
    
    # Calculate D
    for i in range(20):
        if n_counts[i] > 0:
            D[i] = np.sum((miu[i, :] - theta[i])**2) / (n_counts[i]**2)
        else:
            D[i] = 0
    
    # Calculate kesai
    for i in range(20):
        if n_counts[i] > 0:
            kesai[i] = sigma[i] / n_counts[i]
        else:
            kesai[i] = 0
    
    # Calculate covariance
    for i in range(20):
        for j in range(20):
            if i > j:
                if n_counts[i] > 0 and n_counts[j] > 0:
                    cov[i, j] = np.sum((miu[i, :] - theta[i]) * (miu[j, :] - theta[j])) / (n_counts[i] * n_counts[j])
                else:
                    cov[i, j] = 0
            else:
                cov[i, j] = 0
    
    # Assign features
    # First 20: amino acid counts
    feature1[m, :20] = n_counts
    
    # Next 20: kesai values
    feature1[m, 20:40] = kesai
    
    # Next 20: D values
    feature1[m, 40:60] = D
    
    # Next 190: lower triangular of covariance matrix
    mm = 0
    for i in range(20):
        for j in range(20):
            if i > j:
                feature1[m, 60 + mm] = cov[i, j]
                mm += 1

    # Remove rows that are all zeros
    return feature1

for m in range(n1):
    seq = str(s1[m])
    N = len(seq)
    feature1 = np.zeros((n1, 250))
    t0 = np.zeros(N)
    theta = np.zeros(20)
    sigma = np.zeros(20)
    D = np.zeros(20)
    kesai = np.zeros(20)
    cov = np.zeros((20, 20))
    n_counts, t, miu = seq2mats(seq)
    out = anv_process(n_counts, t, miu, N, n1, feature1, t0, theta, sigma, D, kesai, cov)
    out = out[~np.all(out == 0, axis=1)][0]