import pandas as pd
import requests, os
import aiohttp
import asyncio
from tqdm import tqdm
import io
import nest_asyncio
import os

# Apply nest_asyncio to allow running async in notebooks
nest_asyncio.apply()

MEMPROTMD_ROOT_URI = "http://memprotmd.bioch.ox.ac.uk/"

async def get_resid_contacts_async(session, sim_id):
    url = (MEMPROTMD_ROOT_URI 
           + "data/memprotmd/simulations/"
           + sim_id
           + "/files/contacts/by_resid_postprocess.csv")
    
    async with session.get(url) as response:
        if response.status == 200:
            content = await response.read()
            return pd.read_csv(io.StringIO(content.decode('utf-8')))
        else:
            raise Exception(f"HTTP {response.status} for {sim_id}")

async def download_single_pdb_async(session, pdb, outpath="./memprotmd"):
    """Download and save a single PDB file asynchronously"""
    if os.path.exists(outpath + '/' + pdb + '.pkl'):
        return None
    try: 
        df = await get_resid_contacts_async(session, f"{pdb.lower()}_default_dppc")
        df.to_pickle(outpath + "/" + pdb + ".pkl")
    except Exception as e:
        return f"Error: {pdb} - {str(e)}"

async def download_memprotmd_data_csv_async(pdbs, outpath="./memprotmd", max_concurrent=10):
    """Download files asynchronously"""
    # Create output directory if it doesn't exist
    os.makedirs(outpath, exist_ok=True)
    
    # Create a session and semaphore to limit concurrent requests
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [download_single_pdb_async(session, pdb, outpath) for pdb in pdbs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Print results
    for result in results:
        if result == None:
            continue
        print(result)

# Wrapper function for synchronous code
def download_memprotmd_data_pkl(pdbs, outpath="./memprotmd", max_concurrent=10):
    """Synchronous wrapper for async function"""
    asyncio.run(download_memprotmd_data_csv_async(pdbs, outpath, max_concurrent))
