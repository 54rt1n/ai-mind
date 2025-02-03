# mindai/io/jsonl.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import json
from typing import Any, List, Dict

def append_to_jsonl(data: Dict[str, Any], filename: str) -> None:
    """
    Append a single JSON object to a JSONL file.

    Args:
        data (Dict[str, Any]): The JSON object to append.
        filename (str): The path to the JSONL file.

    Returns:
        None
    """
    with open(filename, 'a') as outfile:
        json.dump(data, outfile)
        outfile.write('\n')

def write_jsonl(data: List[Dict[str, Any]], filename: str) -> None:
    """
    Write a list of JSON objects to a JSONL file.

    Args:
        data (List[Dict[str, Any]]): A list of JSON objects to write.
        filename (str): The path to the JSONL file.

    Returns:
        None
    """
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def read_jsonl(filename: str) -> List[Dict[str, Any]]:
    """
    Read a JSONL file and return a list of JSON objects.

    Args:
        filename (str): The path to the JSONL file to read.

    Returns:
        List[Dict[str, Any]]: A list of JSON objects from the file.
    """
    with open(filename, 'r') as f:
        return [json.loads(line) for line in f]