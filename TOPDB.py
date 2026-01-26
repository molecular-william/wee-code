import xml.etree.ElementTree as ET
from collections import namedtuple

DataRow = namedtuple('DataRow', [
    "TOPDB_ID",
    "TOPDB_Type",
    "pdbCode",
    "Membrane",
 ])


def parse_topdb_xml(xml):
    """
    Parse TOPDB XML file using iterparse to extract PDB IDs, Membrane, and Sequence
    """
    # Define namespace
    namespace = {'ns': 'https://topdb.unitmp.org'}
    
    # Use iterparse for memory-efficient parsing
    for event, elem in ET.iterparse(xml, events=('end',)):
        if elem.tag == f'{{{namespace["ns"]}}}TOPDB':
            # Extract TOPDB ID and type
            topdb_id = elem.get('ID')
            topdb_type = elem.get('type')
            
            # Extract membrane information
            membrane_elem = elem.find('.//ns:Membrane', namespace)
            membrane = membrane_elem.text if membrane_elem is not None else None
            
            # Extract PDB entries
            for pdb_elem in elem.findall('.//ns:PDB', namespace):
                pdb_id = pdb_elem.get('ID')
                    
                row = DataRow(
                    TOPDB_ID=topdb_id,
                    TOPDB_Type=topdb_type,
                    pdbCode=pdb_id.upper(),
                    Membrane=membrane,
                )
                yield row
            
            # Clear the element to free memory
            elem.clear()
