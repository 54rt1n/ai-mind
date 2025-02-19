# aim/tool/impl/system.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import os
import subprocess
import psutil
from typing import Dict, Optional, List
from .base import ToolImplementation


class SystemImplementation(ToolImplementation):
    """Implementation of system operations."""
    
    def _get_process_info(self, pid: Optional[int] = None, format: str = "basic") -> List[Dict]:
        """Get process information.
        
        Args:
            pid: Optional specific process ID to query
            format: Output format ('basic' or 'detailed')
            
        Returns:
            List of process information dictionaries
        """
        processes = []
        try:
            if pid:
                proc = psutil.Process(pid)
                processes.append(self._format_process(proc, format))
            else:
                for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 'memory_percent']):
                    processes.append(self._format_process(proc, format))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        return processes

    def _format_process(self, proc: psutil.Process, format: str) -> Dict:
        """Format process information based on requested format."""
        try:
            if format == "basic":
                return {
                    "pid": proc.pid,
                    "name": proc.name(),
                    "status": proc.status()
                }
            else:  # detailed
                return {
                    "pid": proc.pid,
                    "name": proc.name(),
                    "status": proc.status(),
                    "cpu_percent": proc.cpu_percent(),
                    "memory_percent": proc.memory_percent(),
                    "create_time": proc.create_time(),
                    "username": proc.username()
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {"pid": proc.pid, "name": "Access Denied", "status": "unknown"}

    def _get_env_var(self, name: str) -> str:
        """Get environment variable value safely."""
        return os.environ.get(name, "")

    def _run_command(self, command: str, working_dir: Optional[str] = None) -> Dict[str, str]:
        """Run system command safely.
        
        Args:
            command: Command to execute
            working_dir: Optional working directory
            
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

    def execute(self, parameters: Dict[str, str]) -> Dict[str, str]:
        """Execute system operations.
        
        Args:
            parameters: Dictionary containing:
                For process_info:
                    - pid: Process ID (optional)
                    - format: Output format (optional)
                For env_var:
                    - name: Environment variable name
                For run_command:
                    - command: Command to run
                    - working_dir: Working directory (optional)
                
        Returns:
            Dictionary containing operation-specific results
                
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If command execution fails
            PermissionError: If lacking required permissions
        """
        operation = parameters.get("operation", "process_info")
        
        if operation == "process_info":
            pid = parameters.get("pid")
            if pid:
                pid = int(pid)
            format = parameters.get("format", "basic")
            return {
                "processes": self._get_process_info(pid, format)
            }
        elif operation == "env_var":
            if "name" not in parameters:
                raise ValueError("Environment variable name is required")
            return {
                "value": self._get_env_var(parameters["name"])
            }
        elif operation == "run_command":
            if "command" not in parameters:
                raise ValueError("Command is required")
            return self._run_command(
                parameters["command"],
                parameters.get("working_dir")
            )
        else:
            raise ValueError(f"Unknown operation: {operation}") 