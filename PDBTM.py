import pandas as pd
import xml.etree.ElementTree as ET
from collections import namedtuple

DataRow = namedtuple('DataRow', [
     'pdbCode',
     'chain_id',
     'num_tm',
     'chain_type',
     'chain_seq',
     'region_type',
     'prev_type',
     'region_beg',
     'region_end',
     'pdb_beg',
     'pdb_end',
     'N_flank',
     'region_seq',
     'C_flank'
 ])


def get_flanks(sequence: str, region_beg: int, region_end: int, flank_length: int = 5) -> tuple[str, str]:
     n_flank = sequence[:region_beg-1] if region_beg > 0 else ''
     c_flank = sequence[region_end:] if region_end < len(sequence) else ''

     # Truncate flanking regions
     n_flank = n_flank[-flank_length:] if len(n_flank) >= flank_length else n_flank
     c_flank = c_flank[:flank_length] if len(c_flank) >= flank_length else c_flank

     return n_flank, c_flank


def parse_pdbtm_xml(xml: str, flank_length: int = 5, sequence_only: bool = True):
    '''
    Generator to generate data rows from pdbtm xml file.
    '''
    tree = ET.parse(xml)
    root = tree.getroot()
    
    name_spaces = ['{https://pdbtm.unitmp.org}', "{http://pdbtm.enzim.hu}"]
    
    for name_space in name_spaces:
        
        for child in root.iter(f'{name_space}pdbtm'):
            pdbCode = child.get('ID').upper()
            
            for chain in child.findall(f'{name_space}CHAIN'):
                chain_type = chain.get('TYPE')
                chain_id = chain.get('CHAINID')
                num_tm = int(chain.get('NUM_TM'))
                sequence = chain.find(f'{name_space}SEQ').text
                sequence = ''.join(sequence.split()) #formats the sequence
                
                if not sequence_only:
                    region_list = chain.findall(f'{name_space}REGION')
                    for idx, region in enumerate(region_list):
                        region_type = region.get('type')
                        prev_type = region_list[idx-1].get('type') if idx > 0 else 0
                        if region_type != 'H':
                            continue
                        region_beg = int(region.get('seq_beg'))
                        region_end = int(region.get('seq_end'))
                        pdb_beg = int(region.get('pdb_beg'))
                        pdb_end = int(region.get('pdb_end'))
                        region_seq = sequence[region_beg-1 : region_end]
                        if region_type in ['H', 'B']:
                            N_flank, C_flank = get_flanks(sequence, region_beg, region_end, flank_length)
                        else:
                            N_flank, C_flank = '', ''
    
                        row = DataRow(
                            pdbCode=pdbCode,
                            chain_id=chain_id,
                            num_tm=num_tm,
                            chain_type=chain_type,
                            chain_seq=None,
                            region_type=region_type,
                            prev_type=prev_type,
                            region_beg=region_beg,
                            region_end=region_end,
                            pdb_beg=pdb_beg,
                            pdb_end=pdb_end,
                            N_flank=N_flank,
                            region_seq=region_seq,
                            C_flank=C_flank
                        )
                        yield(row)
                else:
                    row = DataRow(
                        pdbCode=pdbCode,
                        chain_id=chain_id,
                        num_tm=num_tm,
                        chain_type=chain_type,
                        chain_seq=sequence,
                        region_type=None,
                        prev_type=None,
                        region_beg=None,
                        region_end=None,
                        pdb_beg=None,
                        pdb_end=None,
                        N_flank=None,
                        region_seq=None,
                        C_flank=None
                    )
                    yield(row)



def get_tmhelix_df(df: pd.DataFrame, align: bool = False, remove_ambiguous: bool = True) -> pd.DataFrame:
    df = df.drop(['chain_type','chain_seq','region_type', 'region_beg', 'region_end', 'pdb_beg','pdb_end'], axis=1)
    df['flanked_region_seq'] = df.N_flank + df.region_seq + df.C_flank
    df = df.drop(['N_flank', 'C_flank'], axis=1)
    
    if remove_ambiguous: #keeps only 1 and 2, intra and extra cellular
        a = ['F', 'H', 'U', 'L','I',0]
        df = df[~df.prev_type.isin(a)]
        df.prev_type = df.prev_type.apply(lambda x: int(x))
        
    df.reset_index(drop=True, inplace=True)
    
    if align:
        df['region_seq_aligned'] = df.apply(lambda row: row.region_seq[::-1] if row.prev_type == 2 else None, axis=1)
        df.region_seq_aligned = df.region_seq_aligned.fillna(df.region_seq)
    
        df['flanked_region_seq_aligned'] = df.apply(lambda row: row.flanked_region_seq[::-1] if row.prev_type == 2 else None, axis=1)
        df.flanked_region_seq_aligned = df.flanked_region_seq_aligned.fillna(df.flanked_region_seq)
        
    return df
    
    
