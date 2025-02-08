# mindai/tool/impl/pipeline.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from typing import Dict, List, Optional, Any
from enum import Enum
import time
import asyncio
from ...pipeline.factory import pipeline_factory
from .base import ToolImplementation


class PipelineStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PENDING = "pending"


class Pipeline:
    """Represents a pipeline instance."""
    
    def __init__(self, pipeline_id: str, parameters: Optional[Dict[str, Any]] = None):
        self.pipeline_id = pipeline_id
        self.job_id = f"job_{int(time.time())}_{pipeline_id}"
        self.parameters = parameters or {}
        self.status = PipelineStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.error = None
        self.result = None

    def to_dict(self) -> Dict:
        """Convert pipeline to dictionary representation."""
        return {
            "pipeline_id": self.pipeline_id,
            "job_id": self.job_id,
            "status": self.status,
            "parameters": self.parameters,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error": self.error,
            "result": self.result
        }

    def start(self):
        """Start the pipeline execution."""
        self.status = PipelineStatus.RUNNING
        self.start_time = time.time()

    def complete(self, result: Any = None):
        """Mark pipeline as completed with optional result."""
        self.status = PipelineStatus.COMPLETED
        self.end_time = time.time()
        self.result = result

    def fail(self, error: str):
        """Mark pipeline as failed with error."""
        self.status = PipelineStatus.FAILED
        self.end_time = time.time()
        self.error = error


class PipelineImplementation(ToolImplementation):
    """Implementation of pipeline operations."""
    
    def __init__(self):
        self._pipelines: Dict[str, Pipeline] = {}
    
    def _list_pipelines(self, status: str = "all") -> List[Dict]:
        """List pipelines with optional status filter."""
        pipelines = []
        for pipeline in self._pipelines.values():
            if status == "all" or pipeline.status == status:
                pipelines.append(pipeline.to_dict())
        return pipelines

    async def _run_pipeline(self, pipeline_id: str, parameters: Optional[Dict] = None) -> Pipeline:
        """Start a new pipeline execution."""
        pipeline = Pipeline(pipeline_id, parameters)
        self._pipelines[pipeline.job_id] = pipeline
        
        try:
            # Start pipeline execution
            pipeline.start()
            
            # Get pipeline function from factory
            pipeline_func = pipeline_factory(pipeline_id)
            if not pipeline_func:
                raise ValueError(f"Unknown pipeline type: {pipeline_id}")
            
            # Execute pipeline
            result = await pipeline_func(pipeline.parameters)
            pipeline.complete(result)
            
        except Exception as e:
            pipeline.fail(str(e))
            raise RuntimeError(f"Pipeline execution failed: {str(e)}")
            
        return pipeline

    def _get_running_pipelines(self) -> List[Dict]:
        """Get all currently running pipelines."""
        return [
            pipeline.to_dict()
            for pipeline in self._pipelines.values()
            if pipeline.status == PipelineStatus.RUNNING
        ]

    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pipeline operations.
        
        Args:
            parameters: Dictionary containing:
                For list:
                    - status: Filter by status (optional)
                For run:
                    - pipeline_id: ID of pipeline to run
                    - parameters: Pipeline parameters (optional)
                For running:
                    (no parameters)
                
        Returns:
            Dictionary containing operation-specific results:
                For list:
                    - pipelines: List of pipeline info
                For run:
                    - job_id: ID of started pipeline job
                    - status: Success message
                For running:
                    - running_pipelines: List of running pipeline info
                
        Raises:
            ValueError: If pipeline_id is invalid
            RuntimeError: If pipeline execution fails
        """
        operation = parameters.get("operation", "list")
        
        if operation == "list":
            status = parameters.get("status", "all")
            return {
                "pipelines": self._list_pipelines(status)
            }
        elif operation == "run":
            if "pipeline_id" not in parameters:
                raise ValueError("pipeline_id is required")
                
            # Run pipeline asynchronously
            pipeline = asyncio.run(self._run_pipeline(
                parameters["pipeline_id"],
                parameters.get("parameters")
            ))
            
            return {
                "job_id": pipeline.job_id,
                "status": f"Successfully started pipeline {parameters['pipeline_id']}",
                "pipeline": pipeline.to_dict()
            }
        elif operation == "running":
            return {
                "running_pipelines": self._get_running_pipelines()
            }
        else:
            raise ValueError(f"Unknown operation: {operation}") 