import math
from statistics import median


hydrophobicity_values = {  # Dictionary mapping amino acids to their hydrophobicity values
        'A': 0.33,
        'R': 1.00,
        'N': 0.43,
        'D': 2.41,
        'C': 0.22,
        'E': 1.61,
        'Q': 0.19,
        'G': 1.14,
        'H': 1.37,
        'I': -0.81,
        'L': -0.69,
        'K': 1.81,
        'M': -0.44,
        'F': -0.58,
        'P': -0.31,
        'S': 0.33,
        'T': 0.11,
        'W': -0.24,
        'Y': 0.23,
        'V': -0.53
    }

# Define hydrophobicity values for each amino acid
hydrophobicity = {
        'A': (0.17, 0.51, 0.33),
        'R': (0.81, 1.81, 1.00),
        'N': (0.42, 0.85, 0.43),
        'D': (1.23, 3.64, 2.41),
        'C': (-0.24, -0.02, 0.22),
        'E': (2.02, 3.63, 1.61),
        'Q': (0.58, 0.77, 0.19),
        'G': (0.01, 1.15, 1.14),
        'H': (0.96, 2.33, 1.37),
        'I': (-0.31, -1.12, -0.81),
        'L': (-0.56, -1.25, -0.69),
        'K': (0.99, 2.80, 1.81),
        'M': (-0.23, -0.67, -0.44),
        'F': (-1.13, -1.71, -0.58),
        'P': (0.45, 0.14, -0.31),
        'S': (0.13, 0.46, 0.33),
        'T': (0.14, 0.25, 0.11),
        'W': (-1.85, -2.09, -0.24),
        'Y': (-0.94, -0.71, 0.23),
        'V': (0.07, -0.46, -0.53),
    }

aa_groups = {  # Define amino acid groups
        'I': 0, 'L': 0, 'V': 0,  # Hydrophobic group (simple TM)
        'C': 1,                   # Hydrophobic
        'A': 2,
        'M': 3,
        'W': 4,
        'F': 5,
        'P': 6,                   # Structural residues
        'G': 7,
        'R': 8,                   # Other individual residues
        'N': 9,
        'D': 10,
        'E': 11,
        'Q': 12,
        'H': 13,
        'K': 14,
        'S': 15,
        'T': 16,
        'Y': 17
    }

def compute_seq_aa_entropy_grp_ivl(sequence):
    """Calculate the entropy of amino acid groups in a protein sequence.
    
    Args:
        sequence: Input protein sequence string
        
    Returns:
        float: Entropy value or -1 if empty sequence
    """
    if not sequence:
        return -1

    # Initialize frequency array with 18 zeros
    freq_array = [0.0] * 18
    valid_length = 0
    
    # Count amino acids by group
    for aa in sequence.upper():
        if aa in aa_groups:
            freq_array[aa_groups[aa]] += 1
            valid_length += 1
    
    if valid_length == 0:
        return -1
    
    # Calculate probabilities and entropy
    entropy = 0.0
    for count in freq_array:
        probability = count / valid_length
        if probability > 0:
            entropy -= probability * math.log2(probability)
    
    return entropy
#################################################################################################


def compute_seq_aa_hydrophobicity(sequence):
    """Calculate various hydrophobicity scales for a protein sequence.
    
    Args:
        sequence: Input protein sequence string
        
    Returns:
        tuple: (octanol_scale, interface_scale, octanol_interface_scale)
               Returns (0.0, 0.0, 0.0) for empty/invalid sequence
    """
    if not sequence:
        return (0.0, 0.0, 0.0)
    
    interface_sum = 0.0
    octanol_sum = 0.0
    octanol_interface_sum = 0.0
    valid_length = 0
    
    for aa in sequence.upper():
        if aa in hydrophobicity:
            interface, octanol, oct_interface = hydrophobicity[aa]
            interface_sum += interface
            octanol_sum += octanol
            octanol_interface_sum += oct_interface
            valid_length += 1
    
    if valid_length == 0:
        return (0.0, 0.0, 0.0)
    
    return (octanol_sum, interface_sum, octanol_interface_sum)
###################################################################################################


def compute_summary_statistics(vector):
    """Calculate summary statistics (mean, std deviation, median, MAD) from a comma-separated string.
    
    Args:
        vector: List of floats
        
    Returns:
        tuple: (center, spread) where center is mean and spread is standard deviation,
               or (-1, -1) for empty/invalid input
    """
    if not vector:
        return (-1, -1)
    
    try:
        # Convert comma-separated string to list of floats
        numarray = vector
    except ValueError:
        return (-1, -1)
    
    if not numarray:
        return (-1, -1)
    
    # Calculate mean and standard deviation
    count = len(numarray)
    totalsum = sum(numarray)
    squaresum = sum(x*x for x in numarray)
    
    center = totalsum / count
    variance = (squaresum / count) - (center ** 2)
    spread = math.sqrt(variance) if variance > 0 else 0.0
    
    # Calculate median
    med = median(numarray)
    
    # Calculate Median Absolute Deviation (MAD)
    abs_deviation = [abs(x - med) for x in numarray]
    mad = median(abs_deviation)
    mad = max(mad, 0.0)  # Ensure non-negative
    
    return (center, spread)
#####################################################################################################################


def get_aa_hydrophobicity(residue):
    """Return the octanol-interface hydrophobicity value for a given amino acid residue.
    
    Args:
        residue: Single character amino acid code (case-insensitive)
        
    Returns:
        float: Hydrophobicity value or 'NaN' for invalid residues
    """
    # Convert to uppercase and get value, return 'NaN' if not found
    return hydrophobicity_values.get(residue.upper(), 0.33)#'NaN')
#########################################################################################################


def compute_zscore(
    entropy,
    hydroscale,
    entropy_center,
    entropy_spread,
    hydroscale_center,
    hydroscale_spread,
    correlation,
    correlation_flag
):
    """Calculate a z-score based on entropy and hydrophobicity scales.
    
    Args:
        entropy: Entropy value
        hydroscale: Hydrophobicity scale value
        entropy_center: Mean of entropy distribution
        entropy_spread: Standard deviation of entropy
        hydroscale_center: Mean of hydrophobicity distribution
        hydroscale_spread: Standard deviation of hydrophobicity
        correlation: Correlation coefficient between entropy and hydrophobicity
        correlation_flag: Whether to use correlation in calculation (1=True, 0=False)
        
    Returns:
        float: Calculated z-score, with sign adjusted based on relationship condition
    """
    # Calculate the terms of the z-score formula
    term1 = ((entropy - entropy_center) ** 2) / (entropy_spread ** 2)
    term3 = ((hydroscale - hydroscale_center) ** 2) / (hydroscale_spread ** 2)
    
    # Include correlation term if flag is set
    if correlation_flag == 1:
        term2 = (2 * correlation * (entropy - entropy_center) * 
                (hydroscale - hydroscale_center)) / (
                    entropy_spread * hydroscale_spread
                )
        zscore = term1 - term2 + term3
    else:
        zscore = term1 + term3
    
    # Slope of the regressed line (note that high hydrophobicity is negative)
    p = 0.436
    
    # Determine sign adjustment based on relationship condition
    if (p * entropy_spread * (hydroscale - hydroscale_center) < 
            -1 * hydroscale_spread * (entropy - entropy_center)):
        return -zscore
    return zscore
#################################################################################################################


def generate_tm_classification(tm_regions, fasta_seq):
    """Classify transmembrane regions based on entropy and hydrophobicity features.
    
    Args:
        tm_regions: Space-separated string of TM regions (e.g., "10,20 30,40")
        fasta_seq: Input protein sequence
        
    Returns:
        list: results_list:
               - results_list contains classification info for each TM region
    """
    results = []
    win_size_x = 12
    win_size_y = 19
    seq_aa = list(fasta_seq)
    # masked_seq_aa = list(fasta_seq.lower())
    
    segments = tm_regions.split()
    
    for segment in segments:
        start, end = map(int, segment.split(','))
        helix_seq = ''.join(seq_aa[start:end+1])
        helix_aa = list(helix_seq)
        
        # Extend the segment if smaller than window sizes
        if len(helix_aa) < max(win_size_x, win_size_y):
            extend_length = max(win_size_x, win_size_y) - len(helix_aa)
            left_pos, right_pos = start, end
            
            for _ in range(extend_length):
                left_pos -= 1
                right_pos += 1
                
                if 0 <= left_pos and right_pos < len(seq_aa):
                    left_hydro = get_aa_hydrophobicity(seq_aa[left_pos])
                    right_hydro = get_aa_hydrophobicity(seq_aa[right_pos])
                    
                    if left_hydro < right_hydro:
                        right_pos -= 1
                    elif left_hydro > right_hydro:
                        left_pos += 1
                elif left_pos < 0 and right_pos < len(seq_aa):
                    left_pos += 1
                elif left_pos >= 0 and right_pos >= len(seq_aa):
                    right_pos -= 1
                
                if (right_pos - left_pos + 1) >= (len(helix_aa) + extend_length):
                    break
            
            helix_seq = ''.join(seq_aa[left_pos:right_pos+1])
            helix_aa = list(helix_seq)
        
        # Calculate entropy vector
        entropy_vector = []
        for x in range(len(helix_aa) - win_size_x + 1):
            subsequence = ''.join(helix_aa[x:x+win_size_x])
            entropy = compute_seq_aa_entropy_grp_ivl(subsequence)
            entropy_vector.append(entropy)
        
        # Calculate hydrophobicity vectors
        octanol_vector, interface_vector, oct2intf_vector = [], [], []
        for y in range(len(helix_aa) - win_size_y + 1):
            subsequence = ''.join(helix_aa[y:y+win_size_y])
            octanol, interface, oct2intf = compute_seq_aa_hydrophobicity(subsequence)
            octanol_vector.append(octanol)
            interface_vector.append(interface)
            oct2intf_vector.append(oct2intf)
        
        # Compute statistics
        ave_entropy, std_entropy = compute_summary_statistics(entropy_vector)
        ave_octanol, std_octanol = compute_summary_statistics(octanol_vector)
        ave_interface, std_interface = compute_summary_statistics(interface_vector)
        ave_oct2intf, std_oct2intf = compute_summary_statistics(oct2intf_vector)
        
        # Compute z-score and classify
        zscore = compute_zscore(
            ave_entropy, ave_oct2intf,
            2.4, 0.30, -0.64, 2.85, 0, 0  # Against functional-TM (UniProt)
        )
        
        if zscore >= -3.29:
            tm_type = "complex"
        elif zscore <= -5.41:
            tm_type = "simple"
        else:
            tm_type = "twilight"

        results.append(tm_type)

    if len(results) == 1:
        return results[0]
    return results
'''
# Format results
formatted_results = (
    f"{helix_seq};{segment};"
    f"{ave_entropy:.2f};{-ave_oct2intf:.2f};"
    f"{zscore:.2f};{tm_type}"
)
results.append(formatted_results)

# Update masked sequence
if tm_type == "simple":
    for j in range(start, end + 1):
        masked_seq_aa[j] = 'X'
elif tm_type in ["complex", "twilight"]:
    for j in range(start, end + 1):
        masked_seq_aa[j] = masked_seq_aa[j].upper()

return results, ''.join(masked_seq_aa)
'''
####################################################################################################################
