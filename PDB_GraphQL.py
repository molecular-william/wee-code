"""
PDB GraphQL Query and Data Processing Module

This module provides functionality to query the Protein Data Bank (PDQL) GraphQL API,
process results asynchronously with batching, and automatically generate schema-aware
data extraction functions. It supports four data types: entries, assemblies,
polymer entities, and polymer entity instances.

Key Features:
- Asynchronous batch processing with configurable batch sizes
- Automatic schema analysis for response data
- Thread-safe execution in various environments (Jupyter, scripts, etc.)
- Progress tracking with tqdm
- Type hints for better code clarity

Potential Issues:
- No error handling for network failures
- Memory issues with very large result sets
- Race conditions in concurrent processing
"""

import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from typing import List, Tuple, Any, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import requests
from pypdb.clients.data.data_types import DataFetcher, DataType
from tqdm import tqdm


# ============================================================================
# MAIN QUERY FUNCTION
# ============================================================================

def pdb_graphql(query_subject: list, query: dict, batch_size: int = 200, max_workers: int = 5) -> pd.DataFrame:
    """
    Main function to query PDB GraphQL API and return results as a DataFrame.
    
    This function handles the entire pipeline: determining data type, creating appropriate
    extractors, processing in batches, and combining results into a DataFrame.
    
    Args:
        query_subject: List of PDB IDs to query (e.g., ['1ABC', '2DEF'])
        query: GraphQL query structure as a dictionary
        batch_size: Number of entries to process in each batch (default: 150)
        max_workers: Maximum number of concurrent worker threads (default: 5)
    
    Returns:
        pd.DataFrame: Processed results with columns derived from the query structure
    
    Raises:
        ValueError: If data type cannot be determined from the query_subject
        RuntimeError: If there are issues with the event loop in async processing
    
    Example:
        >>> query = {'rcsb_id': [], 'exptl': ['method']}
        >>> results = pdb_graphql(['1ABC', '2DEF'], query)
    """
    # Determine data type based on ID format
    # Note: This logic assumes specific ID formats for different data types
    # Potential issue: IDs might not always follow these patterns
    if '_' in query_subject[0]:
        data_type = DataType.POLYMER_ENTITY
        data_type_str = 'polymer_entities'
    elif '.' in query_subject[0]:
        data_type = DataType.POLYMER_ENTITY_INSTANCE
        data_type_str = 'polymer_entity_instances'
    elif '-' in query_subject[0]:
        data_type = DataType.ASSEMBLY
        data_type_str = 'assemblies'
    elif len(query_subject[0]) == 4:
        # WARNING: This assumption might break with non-standard IDs
        data_type = DataType.ENTRY
        data_type_str = 'entries'
    else:
        raise ValueError(f"Cannot determine data type from ID: {query_subject[0]}")
    
    # Create auto-extractor function for the determined data type
    auto_extractor = create_auto_extractor(DataFetcher, data_type, data_type_str)
    
    async def process_single_batch_async(entries_batch: list, query: dict, executor: ThreadPoolExecutor) -> list:
        """
        Process a single batch of entries asynchronously.
        
        Args:
            entries_batch: List of entry IDs in this batch
            query: GraphQL query structure
            executor: ThreadPoolExecutor for running synchronous operations
        
        Returns:
            list: Processed information for entries in the batch
        """
        loop = asyncio.get_event_loop()
        infos = await loop.run_in_executor(
            executor, auto_extractor, entries_batch, query
        )
        return infos
    
    async def process_all_batches_async(entries: list, query: dict, batch_size: int, max_workers: int) -> list:
        """
        Process all batches concurrently with progress tracking.
        
        Args:
            entries: All entry IDs to process
            query: GraphQL query structure
            batch_size: Size of each batch
            max_workers: Maximum concurrent workers
        
        Returns:
            list: Combined results from all batches
        """
        # Create batches - using list comprehension for efficiency
        batches = [
            entries[i * batch_size:(i + 1) * batch_size]
            for i in range((len(entries) + batch_size - 1) // batch_size)
        ]
        
        all_infos = []
        
        # Use ThreadPoolExecutor for synchronous DataFetcher calls
        # WARNING: If max_workers is too high, could overwhelm the API or system
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create tasks for all batches
            tasks = [process_single_batch_async(batch, query, executor) for batch in batches]
            
            # Process with progress bar
            # Note: asyncio.as_completed yields futures in order of completion
            for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing batches"):
                try:
                    infos = await future
                    all_infos.extend(infos)
                except Exception as e:
                    # Log error but continue with other batches
                    print(f"Error processing batch: {e}")
                    continue
        
        return all_infos
    
    # Handle different event loop scenarios
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running (e.g., in Jupyter), run in separate thread
            print("Event loop already running, processing in separate thread...")
            all_infos = run_async_in_thread(
                process_all_batches_async, query_subject, query, batch_size, max_workers
            )
        else:
            # If no running loop, run directly
            all_infos = loop.run_until_complete(
                process_all_batches_async(query_subject, query, batch_size, max_workers)
            )
    except RuntimeError as e:
        # No event loop, create new one
        # Note: asyncio.run() is preferred in Python 3.7+
        all_infos = asyncio.run(
            process_all_batches_async(query_subject, query, batch_size, max_workers)
        )
    except Exception as e:
        # Catch any other exceptions and provide informative error
        print(f"Error during async processing: {e}")
        raise
    
    # Create DataFrame from results
    # OPTIMIZATION: Pre-allocate DataFrame if possible for large datasets
    column_names = column_names_from_query(query)
    df = pd.DataFrame(all_infos, columns=column_names)
    
    return df


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def column_names_from_query(query: dict) -> list:
    """
    Generate column names from the query structure.
    
    The query structure can have fields with subfields. This function flattens
    the structure into dot-separated column names.
    
    Args:
        query: GraphQL query dictionary
    
    Returns:
        list: Flattened column names
    
    Example:
        >>> query = {'field1': [], 'field2': ['sub1', 'sub2']}
        >>> column_names_from_query(query)
        ['field1', 'field2.sub1', 'field2.sub2']
    """
    base_names = list(query.keys())
    output = []
    
    for base_name in base_names:
        if not query[base_name]:  # Empty list means no subfields
            output.append(base_name)
        else:
            for sub_name in query[base_name]:
                output.append(f'{base_name}.{sub_name}')
    
    return output


def run_async_in_thread(async_func, *args) -> Any:
    """
    Run an async function in a separate thread with its own event loop.
    
    This is useful when called from an environment with an already running
    event loop (like Jupyter notebooks).
    
    Args:
        async_func: Async function to execute
        *args: Arguments to pass to the async function
    
    Returns:
        Any: Result from the async function
    
    WARNING: Creating new event loops can have performance implications
    """
    def wrapper():
        """Wrapper to run async function in new event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_func(*args))
        finally:
            loop.close()
    
    # Using a single worker thread to avoid GIL contention issues
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(wrapper)
        return future.result()


# ============================================================================
# SCHEMA ANALYSIS AND AUTOMATIC EXTRACTION
# ============================================================================

class NodeType(Enum):
    """Enum representing different node types in the schema tree."""
    DICT = "dict"
    LIST = "list"
    PRIMITIVE = "primitive"
    NULL = "null"


@dataclass
class SchemaNode:
    """
    Represents a node in the schema tree for GraphQL responses.
    
    Attributes:
        path: String representation of the path to this node
        node_type: Type of node (dict, list, primitive, or null)
        children: Dictionary of child nodes (for dict nodes)
        list_item_schema: Schema for items in a list (for list nodes)
        example_value: Example value encountered during analysis
    """
    path: str
    node_type: NodeType
    children: Dict[str, 'SchemaNode'] = None
    list_item_schema: 'SchemaNode' = None
    example_value: Any = None
    
    def __post_init__(self):
        """Initialize children dictionary if None."""
        if self.children is None:
            self.children = {}


class SchemaAnalyzer:
    """
    Analyzes GraphQL response schema and generates extraction code.
    
    This class inspects sample responses to understand the data structure,
    then generates Python code to extract fields efficiently and safely.
    
    Attributes:
        query: Original GraphQL query structure
        schema: Root SchemaNode representing the response structure
        extraction_rules: Dictionary of rules for extracting each field
    """
    
    def __init__(self, query: Dict[str, List[str]]):
        """
        Initialize SchemaAnalyzer with a query.
        
        Args:
            query: GraphQL query dictionary
        """
        self.query = query
        self.schema = None
        self.extraction_rules = {}
    
    def analyze_response(self, response_data: Dict[str, Any], data_type_str: str = 'entries') -> None:
        """
        Analyze the response data structure and build a schema tree.
        
        Args:
            response_data: Sample response from GraphQL API
            data_type_str: Key in response data containing the entries
            
        Raises:
            ValueError: If response doesn't contain expected structure
        """
        # Validate response structure
        if 'data' not in response_data or data_type_str not in response_data['data']:
            raise ValueError(f"Response must contain 'data.{data_type_str}'")
        
        # Get first entry as sample for schema analysis
        sample_entry = response_data['data'][data_type_str][0]
        self.schema = self._build_schema_tree(sample_entry, "entry")
        self._generate_extraction_rules()
    
    def _build_schema_tree(self, data: Any, path: str) -> SchemaNode:
        """
        Recursively build schema tree from data.
        
        Args:
            data: Data to analyze (dict, list, or primitive)
            path: Current path in the data structure
            
        Returns:
            SchemaNode: Node representing this part of the schema
        """
        if data is None:
            return SchemaNode(path, NodeType.NULL, example_value=None)
        
        elif isinstance(data, dict):
            node = SchemaNode(path, NodeType.DICT, example_value=data)
            for key, value in data.items():
                child_path = f"{path}['{key}']"
                node.children[key] = self._build_schema_tree(value, child_path)
            return node
        
        elif isinstance(data, list):
            node = SchemaNode(path, NodeType.LIST, example_value=data)
            if data:
                # Analyze first item to determine list item schema
                # WARNING: Assumes all list items have same structure
                child_path = f"{path}[0]"
                node.list_item_schema = self._build_schema_tree(data[0], child_path)
            return node
        
        else:  # Primitive types (str, int, float, bool)
            return SchemaNode(path, NodeType.PRIMITIVE, example_value=data)
    
    def _generate_extraction_rules(self) -> None:
        """Generate extraction rules based on query and schema."""
        for field, subfields in self.query.items():
            # Check if field exists in schema
            if field not in self.schema.children:
                self.extraction_rules[field] = {
                    'type': 'missing',
                    'extraction': 'None',
                    'note': f"Field '{field}' not found in response"
                }
                continue
            
            # Generate rule for this field
            field_node = self.schema.children[field]
            self.extraction_rules[field] = self._generate_field_rule(
                field_node, field, subfields
            )
    
    def _generate_field_rule(self, node: SchemaNode, field: str,
                             subfields: List[str]) -> Dict[str, Any]:
        """
        Generate extraction rule for a specific field.
        
        Args:
            node: SchemaNode for this field
            field: Field name
            subfields: List of subfields to extract
            
        Returns:
            dict: Extraction rule with type, extraction code, and notes
        """
        rule = {
            'field': field,
            'node_type': node.node_type.value,
            'extraction': None,
            'safe_extraction': None,
            'required_imports': set(),
            'notes': []
        }
        
        # Handle different node types
        if node.node_type == NodeType.LIST:
            rule = self._handle_list_field(node, field, subfields, rule)
        elif node.node_type == NodeType.DICT:
            rule = self._handle_dict_field(node, field, subfields, rule)
        elif node.node_type == NodeType.PRIMITIVE:
            rule = self._handle_primitive_field(node, field, rule)
        elif node.node_type == NodeType.NULL:
            rule['extraction'] = 'None'
            rule['safe_extraction'] = 'None'
            rule['notes'].append(f"Field '{field}' is always None")
        
        return rule
    
    def _handle_list_field(self, node: SchemaNode, field: str,
                           subfields: List[str], rule: Dict) -> Dict:
        """Handle list-type fields with optional indexing."""
        if not node.list_item_schema:
            rule['extraction'] = f"entry['{field}']"
            rule['safe_extraction'] = f"entry.get('{field}', [])"
            rule['notes'].append("Empty list - no indexing needed")
            return rule
        
        # Check if we need to index into list items
        item_node = node.list_item_schema
        
        if item_node.node_type == NodeType.DICT and subfields:
            # We have a list of dictionaries with subfields - this is where we need to expand
            # Instead of just taking the first item, we need to handle all items
            
            # Check if we should expand (create multiple rows) or aggregate
            # For now, we'll expand by default when we have a list of dicts with subfields
            rule['expand'] = True
            rule['list_item_schema'] = item_node
            
            # Generate extraction for all items in the list
            extractions_per_item = []
            safe_extractions_per_item = []
            
            for subfield in subfields:
                if subfield in item_node.children:
                    # We'll generate code that extracts all items, not just the first
                    extraction = f"[item.get('{subfield}') for item in entry.get('{field}', [])]"
                    safe_extraction = f"[item.get('{subfield}') for item in entry.get('{field}', []) if isinstance(item, dict)]"
                    extractions_per_item.append((subfield, extraction))
                    safe_extractions_per_item.append((subfield, safe_extraction))
                else:
                    rule['notes'].append(f"Subfield '{subfield}' not found in list items")
            
            if extractions_per_item:
                # Store the extraction rules for each subfield
                rule['extraction_dict'] = dict(extractions_per_item)
                rule['safe_extraction_dict'] = dict(safe_extractions_per_item)
                rule['notes'].append("List of dicts detected - will expand to multiple rows")
        else:
            # Just return the list or first item
            if subfields:
                rule['notes'].append(f"List items are not dicts, ignoring subfields: {subfields}")
            rule['extraction'] = f"entry['{field}'][0]" if item_node.node_type != NodeType.LIST else f"entry['{field}']"
            rule['safe_extraction'] = self._generate_safe_extraction(f"entry['{field}']", 0)
        
        rule['notes'].append("List detected")
        rule['required_imports'].add('from typing import Optional')
        
        return rule
    
    def _handle_dict_field(self, node: SchemaNode, field: str,
                           subfields: List[str], rule: Dict) -> Dict:
        """Handle dict-type fields with optional subfield extraction."""
        if not subfields:
            rule['extraction'] = f"entry['{field}']"
            rule['safe_extraction'] = f"entry.get('{field}', {{}})"
            return rule
        
        extractions = []
        safe_extractions = []
        
        for subfield in subfields:
            if subfield in node.children:
                extraction = f"entry['{field}']['{subfield}']"
                safe_extraction = self._generate_safe_extraction(
                    f"entry['{field}']", f"'{subfield}'"
                )
                extractions.append(extraction)
                safe_extractions.append(safe_extraction)
            else:
                rule['notes'].append(f"Subfield '{subfield}' not found")
        
        if len(extractions) == 1:
            rule['extraction'] = extractions[0]
            rule['safe_extraction'] = safe_extractions[0]
        elif extractions:
            # Multiple subfields - return as dict
            extraction_dict = {}
            safe_extraction_dict = {}
            for i, subfield in enumerate(subfields):
                if i < len(extractions):
                    extraction_dict[subfield] = extractions[i]
                    safe_extraction_dict[subfield] = safe_extractions[i]
            
            rule['extraction'] = extraction_dict
            rule['safe_extraction'] = safe_extraction_dict
        
        return rule
    
    def _handle_primitive_field(self, node: SchemaNode, field: str,
                                rule: Dict) -> Dict:
        """Handle primitive-type fields (strings, numbers, booleans)."""
        rule['extraction'] = f"entry['{field}']"
        rule['safe_extraction'] = f"entry.get('{field}')"
        return rule
    
    def _generate_safe_extraction(self, base_path: str,
                                  *accessors: Any) -> str:
        """
        Generate safe extraction code with None checks.
        
        Args:
            base_path: Base variable path (e.g., "entry['field']")
            *accessors: Sequence of accessors (indices or keys)
            
        Returns:
            str: Safe extraction code with conditional checks
        """
        path = base_path
        for accessor in accessors:
            if isinstance(accessor, int):
                path = f"({path}[{accessor}] if {path} and len({path}) > {accessor} else None)"
            else:
                path = f"({path}.get({accessor}) if isinstance({path}, dict) else None)"
        
        return path
    
    def generate_extraction_code(self) -> str:
        """Generate complete extraction function code."""
        
        if not self.extraction_rules:
            raise ValueError("No extraction rules generated. Call analyze_response first.")
        
        # Check if any field needs expansion
        expanding_field = None
        expanding_field_rule = None
        
        for field, rule in self.extraction_rules.items():
            if rule.get('expand', False):
                expanding_field = field
                expanding_field_rule = rule
                break
        
        # Build the function with proper imports
        code_lines = [
            "from typing import Any, Dict, List, Tuple, Optional",
            "",
            "def extract_entry_data(entry: Dict[str, Any]) -> Dict[str, Any]:",
            '    """Extract data from entry based on analyzed schema"""',
        ]
        
        if expanding_field:
            # We need to handle expansion - generate different code
            code_lines = [
                "from typing import Any, Dict, List, Tuple, Optional",
                "",
                "def extract_entry_data(entry: Dict[str, Any]) -> List[Dict[str, Any]]:",
                '    """Extract data from entry with list expansion"""',
                "    results = []",
                "",
                f"    # Get the expanding list: {expanding_field}",
                f"    expanding_list = entry.get('{expanding_field}', [])",
                "    if not expanding_list:",
                "        # Return a single row with None for expanded fields",
                "        result = {}",
            ]
            
            # Add non-expanding fields first
            for field, rule in self.extraction_rules.items():
                if field == expanding_field:
                    continue
                    
                if rule['node_type'] == 'missing':
                    code_lines.append(f"        # {rule['note']}")
                    code_lines.append(f"        result['{field}'] = None")
                else:
                    if isinstance(rule['safe_extraction'], dict):
                        code_lines.append(f"        # Extract {field} (type: {rule['node_type']})")
                        for subfield, extraction in rule['safe_extraction'].items():
                            code_lines.append(f"        result['{field}_{subfield}'] = {extraction}")
                    else:
                        code_lines.append(f"        # Extract {field} (type: {rule['node_type']})")
                        code_lines.append(f"        result['{field}'] = {rule['safe_extraction']}")
            
            # Add placeholders for expanded fields
            if 'safe_extraction_dict' in expanding_field_rule:
                for subfield in expanding_field_rule['safe_extraction_dict']:
                    code_lines.append(f"        result['{expanding_field}_{subfield}'] = None")
            
            code_lines.append("        results.append(result)")
            code_lines.append("    else:")
            code_lines.append("        # Process each item in the expanding list")
            code_lines.append("        for item in expanding_list:")
            code_lines.append("            if not isinstance(item, dict):")
            code_lines.append("                continue")
            code_lines.append("            result = {}")
            
            # Add non-expanding fields (same for all rows)
            for field, rule in self.extraction_rules.items():
                if field == expanding_field:
                    continue
                    
                if rule['node_type'] == 'missing':
                    code_lines.append(f"            # {rule['note']}")
                    code_lines.append(f"            result['{field}'] = None")
                else:
                    if isinstance(rule['safe_extraction'], dict):
                        code_lines.append(f"            # Extract {field} (type: {rule['node_type']})")
                        for subfield, extraction in rule['safe_extraction'].items():
                            code_lines.append(f"            result['{field}_{subfield}'] = {extraction}")
                    else:
                        code_lines.append(f"            # Extract {field} (type: {rule['node_type']})")
                        code_lines.append(f"            result['{field}'] = {rule['safe_extraction']}")
            
            # Add expanded fields from the list item
            if 'safe_extraction_dict' in expanding_field_rule:
                for subfield, extraction in expanding_field_rule['safe_extraction_dict'].items():
                    # Extract from the current item in the list
                    code_lines.append(f"            # Extract {expanding_field}.{subfield} from list item")
                    # Simple extraction from the item dictionary
                    code_lines.append(f"            result['{expanding_field}_{subfield}'] = item.get('{subfield}')")
            
            code_lines.append("            results.append(result)")
            code_lines.append("    ")
            code_lines.append("    return results")
        else:
            # Original non-expanding version
            code_lines.append("    result = {}")
            code_lines.append("")
            
            # Add extraction for each field
            for field, rule in self.extraction_rules.items():
                if rule['node_type'] == 'missing':
                    code_lines.append(f"    # {rule['note']}")
                    code_lines.append(f"    result['{field}'] = None")
                else:
                    if isinstance(rule['safe_extraction'], dict):
                        code_lines.append(f"    # Extract {field} (type: {rule['node_type']})")
                        for subfield, extraction in rule['safe_extraction'].items():
                            code_lines.append(f"    result['{field}_{subfield}'] = {extraction}")
                    else:
                        code_lines.append(f"    # Extract {field} (type: {rule['node_type']})")
                        code_lines.append(f"    result['{field}'] = {rule['safe_extraction']}")
                
                if rule.get('notes'):
                    for note in rule['notes']:
                        code_lines.append(f"    # Note: {note}")
                
                code_lines.append("")
            
            code_lines.append("    return result")
        
        code_lines.append("")
        
        # Update the batch function to handle the new return type
        code_lines.extend([
            "def entry_to_info_batch_auto(entries_batch):",
            '    """Process entries using auto-generated extraction"""',
            "    # Assuming fetcher is defined elsewhere",
            "    fetcher = DataFetcher(entries_batch, DataType.ENTRY)",
            "    query = " + json.dumps(self.query, indent=4),
            "    fetcher.add_property(query)",
            "    fetcher.fetch_data()",
            "    data = fetcher.response['data']['entries']",
            "    ",
            "    results = []",
            "    for entry in data:",
            "        extracted = extract_entry_data(entry)",
            "        # Handle both single dict and list of dicts",
            "        if isinstance(extracted, dict):",
            "            # Single row",
            "            result_tuple = tuple(extracted.values())",
            "            results.append(result_tuple)",
            "        else:",
            "            # List of rows",
            "            for row in extracted:",
            "                result_tuple = tuple(row.values())",
            "                results.append(result_tuple)",
            "    ",
            "    return results"
        ])
        
        return "\n".join(code_lines)
    
    def print_schema_report(self) -> None:
        """Print a human-readable schema report to stdout."""
        print("=" * 80)
        print("SCHEMA ANALYSIS REPORT")
        print("=" * 80)
        print(f"\nQuery analyzed: {self.query}\n")
        
        print("FIELD EXTRACTION RULES:")
        print("-" * 40)
        
        for field, rule in self.extraction_rules.items():
            print(f"\n{field}:")
            print(f"  Type: {rule['node_type']}")
            
            if rule['node_type'] == 'missing':
                print(f"  Status: {rule['note']}")
                continue
            
            if isinstance(rule['extraction'], dict):
                print("  Subfield extractions:")
                for subfield, extraction in rule['extraction'].items():
                    print(f"    {subfield}: {extraction}")
            else:
                print(f"  Direct extraction: {rule['extraction']}")
            
            if isinstance(rule['safe_extraction'], dict):
                print("  Safe subfield extractions:")
                for subfield, extraction in rule['safe_extraction'].items():
                    print(f"    {subfield}: {extraction}")
            else:
                print(f"  Safe extraction: {rule['safe_extraction']}")
            
            if rule.get('notes'):
                print("  Notes:")
                for note in rule['notes']:
                    print(f"    - {note}")
        
        print("\n" + "=" * 80)


def analyze_and_generate(query: Dict[str, List[str]],
                         sample_response: Dict[str, Any] = None,
                         data_type_str: str = 'entries') -> Tuple[str, 'SchemaAnalyzer']:
    """
    Analyze schema and generate extraction code.
    
    Args:
        query: The GraphQL query dictionary
        sample_response: Optional sample response for analysis
        data_type_str: Key in response containing data entries
        
    Returns:
        Tuple of (generated_code, analyzer_object)
    """
    analyzer = SchemaAnalyzer(query)
    
    if sample_response:
        analyzer.analyze_response(sample_response, data_type_str)
    else:
        # In real usage, you would fetch a sample
        print("Warning: No sample response provided. Using mock analysis.")
        # You would typically fetch a sample entry here
    
    # Print report
    analyzer.print_schema_report()
    
    # Generate code
    code = analyzer.generate_extraction_code()
    
    return code, analyzer


def create_auto_extractor(fetcher_class=DataFetcher, data_type_class=DataType.ENTRY, data_type_str='entries'):
    """
    Create an auto-extracting version of entry_to_info_batch
    """
    
    def auto_entry_to_info_batch(entries_batch, query=None):
        """Automatically analyze and extract data for batch of entries"""
        
        if query is None:
            # Use default query from original function
            query = {
                'rcsb_id': [],
                'rcsb_primary_citation': ['pdbx_database_id_DOI'], 
                'exptl': ['method', 'method_details', 'details'],
                'rcsb_entry_info': ['experimental_method'], 
                'struct': ['title'], 
                'struct_keywords': ['text']
            }
        
        # Fetch data for a single entry to analyze schema
        sample_fetcher = fetcher_class([entries_batch[0]], data_type_class)
        sample_fetcher.add_property(query)
        sample_fetcher.fetch_data()
        
        # Analyze schema
        analyzer = SchemaAnalyzer(query)
        analyzer.analyze_response(sample_fetcher.response, data_type_str)
        
        # Generate extractor function dynamically
        namespace = {}
        exec(analyzer.generate_extraction_code(), namespace)
        extractor = namespace['extract_entry_data']
        
        # Fetch full batch
        fetcher = fetcher_class(entries_batch, data_type_class)
        fetcher.add_property(query)
        fetcher.fetch_data()
        data = fetcher.response['data'][data_type_str]
        
        # Extract data using auto-generated function
        results = []
        for entry in data:
            extracted = extractor(entry)
            # Handle both single dict and list of dicts
            if isinstance(extracted, dict):
                # Single row
                result_tuple = tuple(extracted.values())
                results.append(result_tuple)
            else:
                # List of rows (expansion case)
                for row in extracted:
                    result_tuple = tuple(row.values())
                    results.append(result_tuple)
        
        return results
    
    return auto_entry_to_info_batch


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage demonstrating the schema analysis and code generation.
    
    This shows how to use the SchemaAnalyzer to generate extraction code
    from a sample response.
    """
    # Example query (can be any GraphQL query)
    example_query = {
        'rcsb_id': [],
        'rcsb_primary_citation': ['pdbx_database_id_DOI'],
        'exptl': ['method', 'method_details', 'details'],
        'rcsb_entry_info': ['experimental_method'],
        'struct': ['title'],
        'struct_keywords': ['text']
    }
    
    # Mock response data for demonstration
    example_response = {
        'data': {
            'entries': [{
                'rcsb_id': '1ABC',
                'rcsb_primary_citation': {
                    'pdbx_database_id_DOI': '10.2210/pdb1abc/pdb'
                },
                'exptl': [{
                    'method': 'X-RAY DIFFRACTION',
                    'method_details': 'Some details',
                    'details': 'More details'
                }],
                'rcsb_entry_info': {
                    'experimental_method': ['X-RAY DIFFRACTION']
                },
                'struct': {
                    'title': 'Example Structure'
                },
                'struct_keywords': {
                    'text': ['protein', 'enzyme']
                }
            }]
        }
    }
    
    # Analyze and generate code
    print("Running schema analysis example...")
    generated_code, analyzer = analyze_and_generate(example_query, example_response)
    
    print("\n" + "=" * 80)
    print("GENERATED EXTRACTION CODE:")
    print("=" * 80)
    print(generated_code)
