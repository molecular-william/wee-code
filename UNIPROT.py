import requests, sys, json
from io import StringIO
import pandas as pd
import re
from tqdm import tqdm


WEBSITE_API = "https://rest.uniprot.org/"
PROTEINS_API = "https://www.ebi.ac.uk/proteins/api"


#helper function to download data
def get_url(url, **kwargs):
    response = requests.get(url, **kwargs);
    if not response.ok:
        print(response.text)
        response.raise_for_status()
        sys.exit()
    return response


def clean_up_subcellular_location(string: str) -> str:
    """Given string of Subcellular location [CC] from Uniprot, return cleaned version"""
    if string == '':
        return
    string = string[22:]
    string = string.split(' Note=')[0]
    string = re.sub(r'\s\{.+\}', "", string)
    string = re.sub(r'; Multi-pass membrane protein\.\s*', ". ", string)
    string = re.sub(r'; Single-pass membrane protein[\.\;]\s*', ". ", string)
    string = re.sub(r'; Single-pass type II membrane protein[\.\;]\s*', ". ", string)
    string = re.sub(r'; Single-pass type I membrane protein[\.\;]\s*', ". ", string)
    string = re.sub(r'\[.+\]\:\s', "", string)
    return string.rstrip()


def clean_up_function(string: str) -> str:
    """Given string of Function [CC] from Uniprot, return cleaned version"""
    if string == '':
        return
    string = re.sub(r'\s\{ECO:.+?\}\.', "", string)
    string = re.sub(r'\(PubMed:.+?\)', "", string)
    return string.rstrip()


def clean_up_domain(string: str) -> str:
    """Given string of Domain [FT] from Uniprot, return cleaned version"""
    if string == '':
        return
    string = re.sub(r'(\/e.+?\d\"\;?\s?)', "", string)
    string = re.sub(r'DO.+?\;\s', "", string)
    string = string.replace('"', '')
    string = re.sub(r"\/n.+?\=", "", string)
    return string.rstrip()[:-1]


def clean_up_seq_sim(string: str) -> str:
    """Given string of Sequence Similarity from Uniprot, return cleaned version"""
    if string == '':
        return
    string = re.sub(r'SIM.+the\s', "", string)
    string = string[0].upper() + string[1:]
    string = re.sub(r'\s\{ECO:.+?\}\.', "", string)
    return string.rstrip()


def get_uniprot_info(uniprot_entries: pd.Series) -> pd.DataFrame:
    """Given pd.Series of Uniprot accession codes, return Similarity, Subcellular Location, Function and Domains"""
    unique_entries = uniprot_entries.unique()
    length = len(unique_entries)
    batch_size = 1000

    for i in tqdm(range(length // batch_size + 1)):
        joined = ','.join(unique_entries[i * batch_size : (i+1) * batch_size])
        r = get_url(f'{WEBSITE_API}/uniprotkb/accessions?accessions={joined}&fields=accession,cc_similarity,cc_subcellular_location,cc_function,ft_domain&format=tsv')
        df = pd.read_csv(StringIO(r.text), sep='\t')
        if i == 0:
            final_df = df
        else:
            final_df = pd.concat([final_df, df]).reset_index(drop=True)
    return final_df


def values_from_list_of_dict(ls):
    for i in ls:
        yield list(i.values())[0]
