import concurrent.futures
from typing import List, Dict
from tqdm import tqdm
from emdb.client import EMDB


def fetch_single_emdb_sync(emdb_id: str, client):
    """Synchronous version for use with ThreadPoolExecutor"""
    try:
        emdb_entry = client.get_entry(emdb_id)
        specimen_preparation = emdb_entry.structure_determination_list[0]['specimen_preparation_list']['specimen_preparation'][0]
        admin = emdb_entry.admin
        components = None
        details = None  # some info can be found in the details and not mentioned in the components
        if "buffer" in specimen_preparation.keys():
            buffer = specimen_preparation["buffer"]
            components = parse_buffer(buffer)
        if "details" in specimen_preparation.keys():
            details = specimen_preparation["details"]
        keywords, title = parse_admin(admin)
        return emdb_id, components, details, keywords, title
        
    except Exception as e:
        print(f"Error processing {emdb_id}: {str(e)}")
        return emdb_id, None, None, None, None

def fetch_emdb_entries_parallel(emdb_ids: List[str], client, max_workers: int = 5):
    """Fetch EMDB entries in parallel using ThreadPoolExecutor"""
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_emdb = {
            executor.submit(fetch_single_emdb_sync, emdb_id, client): emdb_id 
            for emdb_id in emdb_ids
        }
        
        # Collect results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_emdb), total=len(emdb_ids)):
            emdb_id = future_to_emdb[future]
            try:
                result_emdb_id, components, details, keywords, title = future.result()
                results.append((result_emdb_id, components, details, keywords, title))
            except Exception as e:
                print(f"Exception for {emdb_id}: {e}")
    
    return results


def parse_buffer(buffer: dict):
    pH = buffer["ph"] if "ph" in buffer.keys() else None
    pH_str = "pH " + str(pH) 
    if "component" in buffer.keys():
        components = buffer["component"]
        components_list = []
        for component in components:
            conc_info = component["concentration"] if "concentration" in component.keys() else ""
            conc = conc_info["valueOf_"] + ' ' + conc_info["units"] if conc_info else ""
            name = None
            if "name" in component.keys():
                if "detergent" in component["name"].lower():  # Example: "LMNG detergent"
                    name = component["formula"]
                else:
                    name = component["name"]
            else:
                name = component["formula"]
            component_str = f"{name}: {conc}"
            components_list.append(component_str)
        components_list.append(pH_str)
        components_str = " + ".join(components_list)
    else:
        components_str = pH_str

    return components_str


def parse_admin(admin: dict):
    keywords = admin["keywords"] if "keywords" in admin.keys() else None
    return keywords, admin["title"]
