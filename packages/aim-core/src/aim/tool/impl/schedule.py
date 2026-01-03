# aim/tool/impl/schedule.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from typing import Dict, Any, List, Optional
from .base import ToolImplementation
import datetime
import uuid


class ScheduleTool(ToolImplementation):
    """Implementation of calendar/schedule operations."""
    
    # Mock storage for events
    _events = {
        "evt_12345": {
            "id": "evt_12345",
            "title": "Team Meeting",
            "start_time": "2025-03-15T14:00:00",
            "end_time": "2025-03-15T15:00:00",
            "description": "Weekly team sync",
            "location": "Conference Room A",
            "tags": ["work", "meeting"]
        },
        "evt_67890": {
            "id": "evt_67890",
            "title": "Doctor Appointment",
            "start_time": "2025-03-20T10:30:00",
            "end_time": "2025-03-20T11:30:00",
            "description": "Annual checkup",
            "location": "Health Clinic",
            "tags": ["personal", "health"]
        }
    }
    
    def schedule_get(self, start_date: Optional[str] = None, end_date: Optional[str] = None, 
                    filter: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """View calendar/schedule events.
        
        Args:
            start_date: Start date (ISO format YYYY-MM-DD)
            end_date: End date (ISO format YYYY-MM-DD)
            filter: Filter by event type or tag
            **kwargs: Additional parameters that will be ignored
            
        Returns:
            Dictionary with events
        """
        events = list(self._events.values())
        
        # Apply date filters
        if start_date:
            start = datetime.datetime.fromisoformat(start_date)
            events = [e for e in events if datetime.datetime.fromisoformat(e["start_time"][:10]) >= start]
            
        if end_date:
            end = datetime.datetime.fromisoformat(end_date)
            events = [e for e in events if datetime.datetime.fromisoformat(e["start_time"][:10]) <= end]
            
        # Apply tag/type filter
        if filter:
            filtered_events = []
            for event in events:
                # Check if filter matches tags
                if "tags" in event and filter in event["tags"]:
                    filtered_events.append(event)
                # Check if filter is in title or description
                elif (filter.lower() in event["title"].lower() or 
                      (event.get("description") and filter.lower() in event["description"].lower())):
                    filtered_events.append(event)
            events = filtered_events
            
        return {
            "events": events,
            "count": len(events)
        }
    
    def schedule_add(self, title: str, start_time: str, end_time: Optional[str] = None, 
                    description: Optional[str] = None, location: Optional[str] = None, 
                    tags: Optional[List[str]] = None, **kwargs: Any) -> Dict[str, Any]:
        """Add an event to the schedule.
        
        Args:
            title: Event title
            start_time: Start time (ISO format YYYY-MM-DDTHH:MM:SS)
            end_time: End time (ISO format YYYY-MM-DDTHH:MM:SS)
            description: Event description
            location: Event location
            tags: Event tags
            **kwargs: Additional parameters that will be ignored
            
        Returns:
            Dictionary with event details
            
        Raises:
            ValueError: If start time is invalid
        """
        # Generate event ID
        event_id = f"evt_{str(uuid.uuid4())[:8]}"
        
        # Calculate end time if not provided (1 hour later)
        if not end_time:
            start = datetime.datetime.fromisoformat(start_time)
            end = start + datetime.timedelta(hours=1)
            end_time = end.isoformat()
            
        # Create event
        event = {
            "id": event_id,
            "title": title,
            "start_time": start_time,
            "end_time": end_time
        }
        
        # Add optional fields
        if description:
            event["description"] = description
        if location:
            event["location"] = location
        if tags:
            event["tags"] = tags
            
        # Store event
        self._events[event_id] = event
        
        return {
            "status": "success",
            "event": event
        }
    
    def schedule_update(self, event_id: str, title: Optional[str] = None, 
                       start_time: Optional[str] = None, end_time: Optional[str] = None, 
                       description: Optional[str] = None, location: Optional[str] = None, 
                       **kwargs: Any) -> Dict[str, Any]:
        """Update an existing event.
        
        Args:
            event_id: ID of the event to update
            title: New event title
            start_time: New start time (ISO format)
            end_time: New end time (ISO format)
            description: New event description
            location: New event location
            **kwargs: Additional parameters that will be ignored
            
        Returns:
            Dictionary with updated event
            
        Raises:
            ValueError: If event doesn't exist
        """
        if event_id not in self._events:
            return {
                "status": "error",
                "message": f"Event not found: {event_id}"
            }
            
        event = self._events[event_id]
        
        # Update fields if provided
        if title:
            event["title"] = title
        if start_time:
            event["start_time"] = start_time
        if end_time:
            event["end_time"] = end_time
        if description:
            event["description"] = description
        if location:
            event["location"] = location
            
        # Store updated event
        self._events[event_id] = event
        
        return {
            "status": "success",
            "event": event
        }
    
    def schedule_remove(self, event_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Remove an event from the schedule.
        
        Args:
            event_id: ID of the event to remove
            **kwargs: Additional parameters that will be ignored
            
        Returns:
            Dictionary with removal status
        """
        if event_id in self._events:
            removed_event = self._events.pop(event_id)
            return {
                "status": "success",
                "message": f"Event '{removed_event['title']}' removed",
                "event_id": event_id
            }
        else:
            return {
                "status": "error",
                "message": f"Event not found: {event_id}"
            }
    
    def execute(self, function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute schedule operations.
        
        Args:
            function_name: Name of the function to execute
            parameters: Dictionary of parameter names and values
            
        Returns:
            Dictionary containing operation-specific results
            
        Raises:
            ValueError: If function is unknown or parameters are invalid
        """
        if function_name == "schedule_get":
            return self.schedule_get(
                start_date=parameters.get("start_date"),
                end_date=parameters.get("end_date"),
                filter=parameters.get("filter"),
                **parameters
            )
            
        elif function_name == "schedule_add":
            if "title" not in parameters:
                raise ValueError("Title parameter is required")
            if "start_time" not in parameters:
                raise ValueError("Start time parameter is required")
                
            return self.schedule_add(
                title=parameters["title"],
                start_time=parameters["start_time"],
                end_time=parameters.get("end_time"),
                description=parameters.get("description"),
                location=parameters.get("location"),
                tags=parameters.get("tags"),
                **parameters
            )
            
        elif function_name == "schedule_update":
            if "event_id" not in parameters:
                raise ValueError("Event ID parameter is required")
                
            return self.schedule_update(
                event_id=parameters["event_id"],
                title=parameters.get("title"),
                start_time=parameters.get("start_time"),
                end_time=parameters.get("end_time"),
                description=parameters.get("description"),
                location=parameters.get("location"),
                **parameters
            )
            
        elif function_name == "schedule_remove":
            if "event_id" not in parameters:
                raise ValueError("Event ID parameter is required")
                
            return self.schedule_remove(
                event_id=parameters["event_id"],
                **parameters
            )
            
        else:
            raise ValueError(f"Unknown function: {function_name}") 