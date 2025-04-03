# aim/tool/impl/file_ops.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import os
import shutil
import glob
from typing import Dict, Any, List, Optional
from .base import ToolImplementation


class FileTool(ToolImplementation):
    """Implementation of file system operations."""
    
    def file_read(self, path: str, encoding: str = "utf-8", **kwargs: Any) -> Dict[str, Any]:
        """Read contents of a file.
        
        Args:
            path: Path to the file to read
            encoding: File encoding
            **kwargs: Additional parameters that will be ignored
            
        Returns:
            Dictionary with file content
            
        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If permission denied
        """
        try:
            with open(path, 'r', encoding=encoding) as f:
                content = f.read()
            
            return {
                "content": content,
                "path": path
            }
        except UnicodeDecodeError:
            # Fall back to binary mode if text decoding fails
            with open(path, 'rb') as f:
                binary_content = f.read()
            return {
                "content": "<binary content>",
                "binary": True,
                "size": len(binary_content),
                "path": path
            }
    
    def file_write(self, path: str, content: str, encoding: str = "utf-8", **kwargs: Any) -> Dict[str, Any]:
        """Write content to a file.
        
        Args:
            path: Path to write the file to
            content: Content to write to the file
            encoding: File encoding
            **kwargs: Additional parameters that will be ignored
            
        Returns:
            Dictionary with status information
            
        Raises:
            PermissionError: If permission denied
            IOError: If write fails
        """
        # Create directory structure if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)
        
        return {
            "status": "success",
            "path": path,
            "bytes_written": len(content.encode(encoding))
        }
    
    def file_list(self, path: str, pattern: Optional[str] = None, **kwargs: Any) -> Dict[str, List[str]]:
        """List contents of a directory.
        
        Args:
            path: Directory path to list
            pattern: Optional glob pattern to filter files
            **kwargs: Additional parameters that will be ignored
            
        Returns:
            Dictionary with file list
            
        Raises:
            FileNotFoundError: If directory doesn't exist
            PermissionError: If permission denied
        """
        if pattern:
            files = glob.glob(os.path.join(path, pattern))
        else:
            files = [os.path.join(path, f) for f in os.listdir(path)]
        
        # Separate directories and files
        dirs = [f for f in files if os.path.isdir(f)]
        regular_files = [f for f in files if os.path.isfile(f)]
        
        return {
            "directories": dirs,
            "files": regular_files,
            "path": path
        }
    
    def file_delete(self, path: str, **kwargs: Any) -> Dict[str, Any]:
        """Delete a file.
        
        Args:
            path: Path of the file to delete
            **kwargs: Additional parameters that will be ignored
            
        Returns:
            Dictionary with deletion status
            
        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If permission denied
        """
        if os.path.isfile(path):
            os.remove(path)
            status = "success"
        else:
            status = "error"
            error = "File not found"
        
        return {
            "status": status,
            "path": path,
            "error": error if status == "error" else None
        }
    
    def file_move(self, source: str, destination: str, **kwargs: Any) -> Dict[str, Any]:
        """Move or rename a file.
        
        Args:
            source: Source file path
            destination: Destination file path
            **kwargs: Additional parameters that will be ignored
            
        Returns:
            Dictionary with move status
            
        Raises:
            FileNotFoundError: If source file doesn't exist
            PermissionError: If permission denied
        """
        # Create destination directory if it doesn't exist
        dest_dir = os.path.dirname(os.path.abspath(destination))
        os.makedirs(dest_dir, exist_ok=True)
        
        shutil.move(source, destination)
        
        return {
            "status": "success",
            "source": source,
            "destination": destination
        }
    
    def execute(self, function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute file system operations.
        
        Args:
            function_name: Name of the function to execute
            parameters: Dictionary of parameter names and values
            
        Returns:
            Dictionary containing operation-specific results
            
        Raises:
            ValueError: If parameters are invalid
            FileNotFoundError: If file doesn't exist
            PermissionError: If permission denied
        """
        if function_name == "file_read":
            if "path" not in parameters:
                raise ValueError("File path is required")
            return self.file_read(
                path=parameters["path"],
                encoding=parameters.get("encoding", "utf-8"),
                **parameters
            )
            
        elif function_name == "file_write":
            if "path" not in parameters:
                raise ValueError("File path is required")
            if "content" not in parameters:
                raise ValueError("Content is required")
            return self.file_write(
                path=parameters["path"],
                content=parameters["content"],
                encoding=parameters.get("encoding", "utf-8"),
                **parameters
            )
            
        elif function_name == "file_list":
            if "path" not in parameters:
                raise ValueError("Directory path is required")
            return self.file_list(
                path=parameters["path"],
                pattern=parameters.get("pattern"),
                **parameters
            )
            
        elif function_name == "file_delete":
            if "path" not in parameters:
                raise ValueError("File path is required")
            return self.file_delete(
                path=parameters["path"],
                **parameters
            )
            
        elif function_name == "file_move":
            if "source" not in parameters:
                raise ValueError("Source file path is required")
            if "destination" not in parameters:
                raise ValueError("Destination file path is required")
            return self.file_move(
                source=parameters["source"],
                destination=parameters["destination"],
                **parameters
            )
            
        else:
            raise ValueError(f"Unknown function: {function_name}") 