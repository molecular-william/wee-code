import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
from collections import namedtuple


def parse_mpstruc(xml):
    DataRow = namedtuple('DataRow', [
        'group','subgroup',
        'main_struc','main_name',
        'main_res','main_species','main_taxdom',
        'member_struc','member_name',
        'member_species','member_taxdom','member_res', 
    ])
    tree = ET.parse(xml)
    root = tree.getroot()
    #ns = '{https://topdb.unitmp.org}'
    for group in root.find('groups').findall('group'):
        group_name = group.find('name').text

        for subgroup in group.find('subgroups').findall('subgroup'):
            subgroup_name = subgroup.find('name').text

            for protein in subgroup.find('proteins').findall('protein'):
                main_struc = protein.find('pdbCode').text
                main_name = protein.find('name').text
                main_res = protein.find('resolution').text
                main_species = protein.find('species').text
                main_taxdom = protein.find('taxonomicDomain').text

                if protein.find('memberProteins').findall('memberProtein'):
                    for memberProtein in protein.find('memberProteins').findall('memberProtein'):
                        member_struc = memberProtein.find('pdbCode').text
                        member_name = memberProtein.find('name').text
                        member_res = protein.find('resolution').text
                        member_species = protein.find('species').text
                        member_taxdom = protein.find('taxonomicDomain').text
                        
                        yield DataRow(group=group_name,subgroup=subgroup_name,main_struc=main_struc,
                                 main_name=main_name,main_res=main_res,main_species=main_species,
                                 main_taxdom=main_taxdom,member_struc=member_struc,member_name=member_name,
                                 member_res=member_res,member_species=member_species,member_taxdom=member_taxdom)
                
                else:
                    member_struc, member_name, member_res = None, None, None
                    member_species, member_taxdom = None, None
                
                    yield DataRow(group=group_name,subgroup=subgroup_name,main_struc=main_struc,
                                 main_name=main_name,main_res=main_res,main_species=main_species,
                                 main_taxdom=main_taxdom,member_struc=member_struc,member_name=member_name,
                                 member_res=member_res,member_species=member_species,member_taxdom=member_taxdom)


def format_mpstruc_df(df: pd.DataFrame) -> pd.DataFrame:
    unique_strucs = df.main_struc.unique()
    for unique_struc in unique_strucs:
        idx = (df.main_struc == unique_struc).idxmax()  # index of first occurrence
        if df.iloc[idx].member_struc:  # if the entry isn't None
            new_row = df.iloc[idx].to_dict()
            new_row['member_struc'] = new_row['main_struc']
            new_row['member_name'] = new_row['main_name']
            new_row['member_species'] = new_row['main_species']
            new_row['member_taxdom'] = new_row['main_taxdom']
            new_row['member_res'] = new_row['main_res']
            df.loc[idx-0.5] = tuple(new_row.values())  # insert the row before the idx'th row
            df = df.sort_index().reset_index(drop=True)
        else:
            df.iloc[idx].member_struc = df.iloc[idx].main_struc
            df.iloc[idx].member_name = df.iloc[idx].main_name
            df.iloc[idx].member_species = df.iloc[idx].main_species
            df.iloc[idx].member_taxdom = df.iloc[idx].main_taxdom
            df.iloc[idx].member_res = df.iloc[idx].main_res
    return df
