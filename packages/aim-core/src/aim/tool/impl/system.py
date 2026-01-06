# aim/tool/impl/system.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import os
import subprocess
import sys
import tempfile
from typing import Dict, Any, Optional
from .base import ToolImplementation


class SystemTool(ToolImplementation):
    """Implementation of system operations."""
    
    def system_run_command(self, command: str, working_dir: str = ".", **kwargs: Any) -> Dict[str, Any]:
        """Run a bash command.
        
        Args:
            command: The command to run
            working_dir: Working directory for command execution
            **kwargs: Additional parameters that will be ignored
            
        Returns:
            Dictionary with command output and exit code
            
        Raises:
            RuntimeError: If command execution fails
            PermissionError: If command is not allowed
        """
        # Basic security check - prevent dangerous commands
        dangerous_commands = ['rm -rf', 'mkfs', 'dd', '>', '>>', '|', ';']
        if any(cmd in command for cmd in dangerous_commands):
            raise PermissionError(f"Command contains dangerous operations: {command}")

        try:
            process = subprocess.run(
                command,
                shell=True,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=30  # Prevent long-running commands
            )
            return {
                "output": process.stdout + process.stderr,
                "exit_code": process.returncode
            }
        except subprocess.TimeoutExpired:
            raise RuntimeError("Command timed out")
        except subprocess.SubprocessError as e:
            raise RuntimeError(f"Command execution failed: {str(e)}")

    def system_run_python(self, code: str, working_dir: str = ".", **kwargs: Any) -> Dict[str, Any]:
        """Execute a Python script.
        
        Args:
            code: Python code to execute
            working_dir: Working directory for execution
            **kwargs: Additional parameters that will be ignored
            
        Returns:
            Dictionary with execution result
            
        Raises:
            RuntimeError: If execution fails
        """
        # Create a temporary file for the code
        try:
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp:
                temp_path = temp.name
                temp.write(code)
                
            # Run the script with captured output
            result = subprocess.run(
                [sys.executable, temp_path],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "output": result.stdout,
                "error": result.stderr,
                "exit_code": result.returncode
            }
        except Exception as e:
            raise RuntimeError(f"Python execution failed: {str(e)}")
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_path)
            except:
                pass

    def system_env_var(self, var_name: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """Get environment variable values.
        
        Args:
            var_name: Specific environment variable to get (optional)
            **kwargs: Additional parameters that will be ignored
            
        Returns:
            Dictionary with variable values
        """
        if var_name:
            return {
                "value": os.environ.get(var_name, "")
            }
        else:
            # Return all environment variables
            return {
                "variables": dict(os.environ)
            }

    def execute(self, function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute system operations.
        
        Args:
            function_name: Name of the function to execute
            parameters: Dictionary of parameter names and values
                
        Returns:
            Dictionary containing operation-specific results
                
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If command execution fails
            PermissionError: If lacking required permissions
        """
        if function_name == "system_run_command":
            if "command" not in parameters:
                raise ValueError("Command is required")
            return self.system_run_command(
                command=parameters["command"],
                working_dir=parameters.get("working_dir", "."),
                **parameters
            )
        elif function_name == "system_run_python":
            if "code" not in parameters:
                raise ValueError("Python code is required")
            return self.system_run_python(
                code=parameters["code"],
                working_dir=parameters.get("working_dir", "."),
                **parameters
            )
        elif function_name == "system_env_var":
            return self.system_env_var(
                var_name=parameters.get("var_name"),
                **parameters
            )
        else:
            raise ValueError(f"Unknown function: {function_name}") 