"""
Human-in-the-Loop (HITL) Interface for the MAS framework.
Provides transport-agnostic interface for user interactions with audit logging.
"""

import logging
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

class HitlInterface(ABC):
    """Abstract base class for Human-in-the-Loop interactions."""
    
    def __init__(self, audit_log_path: Optional[str] = None):
        """
        Initialize HITL interface with optional audit logging.
        Args:
            audit_log_path: Path to audit log file. If None, uses default location.
        """
        self.audit_log_path = audit_log_path or "logs/hitl_audit.json"
        self._ensure_audit_dir()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _ensure_audit_dir(self):
        """Ensure audit log directory exists."""
        os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)
    
    def _log_audit(self, event_type: str, context: Dict[str, Any], 
                   user_input: Optional[str] = None, system_response: Optional[str] = None):
        """Log HITL interaction to audit file."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event_type": event_type,
            "context": context,
            "user_input": user_input,
            "system_response": system_response
        }
        
        # Append to audit log file
        try:
            with open(self.audit_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(audit_entry) + '\n')
        except Exception as e:
            logging.warning(f"Failed to write audit log: {e}")
    
    @abstractmethod
    def prompt_user(self, message: str, options: Optional[List[str]] = None, 
                   multi_select: bool = False) -> Union[str, List[str]]:
        """
        Prompt user for input.
        Args:
            message: The prompt message
            options: List of available options (for selection)
            multi_select: Whether multiple selections are allowed
        Returns:
            User input (string or list of strings)
        """
        pass
    
    @abstractmethod
    def show_info(self, message: str, data: Optional[Any] = None):
        """
        Show information to user.
        Args:
            message: Information message
            data: Optional data to display (DataFrame, dict, etc.)
        """
        pass
    
    def prompt_with_audit(self, message: str, options: Optional[List[str]] = None, 
                         multi_select: bool = False, context: Optional[Dict[str, Any]] = None) -> Union[str, List[str]]:
        """
        Prompt user with audit logging.
        Args:
            message: The prompt message
            options: List of available options
            multi_select: Whether multiple selections are allowed
            context: Additional context for audit logging
        Returns:
            User input
        """
        context = context or {}
        context.update({
            "prompt_message": message,
            "options": options,
            "multi_select": multi_select
        })
        
        # Log the prompt
        self._log_audit("prompt", context)
        
        # Get user input
        user_input = self.prompt_user(message, options, multi_select)
        
        # Log the response
        self._log_audit("response", context, user_input=str(user_input))
        
        return user_input
    
    def show_info_with_audit(self, message: str, data: Optional[Any] = None, 
                           context: Optional[Dict[str, Any]] = None):
        """
        Show information with audit logging.
        Args:
            message: Information message
            data: Optional data to display
            context: Additional context for audit logging
        """
        context = context or {}
        context.update({
            "info_message": message,
            "data_type": type(data).__name__ if data is not None else None
        })
        
        # Log the info display
        self._log_audit("info_display", context, system_response=message)
        
        # Show the information
        self.show_info(message, data)

    def show_warning(self, message: str, data: Optional[Any] = None):
        """
        Display warning information. By default, delegates to show_info with a warning prefix.
        Subclasses can override for custom rendering.
        """
        warning_message = f"⚠️ {message}"
        self.show_info(warning_message, data)

    def show_warning_with_audit(self, message: str, data: Optional[Any] = None,
                                context: Optional[Dict[str, Any]] = None):
        """
        Show warning information with audit logging.
        """
        context = context or {}
        context.update({
            "warning_message": message,
            "data_type": type(data).__name__ if data is not None else None
        })

        self._log_audit("warning_display", context, system_response=message)
        self.show_warning(message, data)


class CliHitlInterface(HitlInterface):
    """CLI-based HITL interface implementation."""
    
    def prompt_user(self, message: str, options: Optional[List[str]] = None,
                   multi_select: bool = False) -> Union[str, List[str]]:
        """Prompt user via command line interface."""
        print(f"\n{message}")

        if options:
            print("Available options:")
            for i, option in enumerate(options):
                print(f"  [{i}] {option}")

            if os.getenv("HITL_AUTO") == "1":
                if multi_select:
                    logging.info(f"[HITL_AUTO] auto-selecting all options: {options}")
                    return list(options)
                logging.info(f"[HITL_AUTO] auto-selecting option [0]: {options[0]}")
                return options[0]

            if multi_select:
                while True:
                    try:
                        indices_input = input("Enter comma-separated numbers for your choices: ").strip()
                        if not indices_input:
                            return []
                        indices = [int(i.strip()) for i in indices_input.split(',')]
                        if all(0 <= i < len(options) for i in indices):
                            return [options[i] for i in indices]
                        else:
                            print("Invalid selection. Please try again.")
                    except ValueError:
                        print("Invalid input. Please enter comma-separated numbers.")
            else:
                while True:
                    try:
                        index = int(input("Enter the number of your choice: ").strip())
                        if 0 <= index < len(options):
                            return options[index]
                        else:
                            print("Invalid selection. Please try again.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
        else:
            return input("Your input: ").strip()
    
    def show_info(self, message: str, data: Optional[Any] = None):
        """Show information via command line interface."""
        print(f"\n{message}")
        if data is not None:
            if hasattr(data, 'to_string'):
                print(data.to_string())
            elif isinstance(data, (dict, list)):
                print(json.dumps(data, indent=2, default=str))
            else:
                print(str(data))


class WebHitlInterface(HitlInterface):
    """Web-based HITL interface backed by thread-safe queues.

    Wires into the Streamlit webapp via two queues:
      - event_queue:    backend pushes events (info/warning/prompt) for the UI
      - response_queue: UI pushes user responses back to the workflow

    When no queues are wired (queues=None), behaves like CliHitlInterface
    so existing CLI usage keeps working.
    """

    def __init__(self, audit_log_path: Optional[str] = None,
                 event_queue=None, response_queue=None, **kwargs):
        super().__init__(audit_log_path)
        self.event_queue = event_queue
        self.response_queue = response_queue
        self._cli_fallback = CliHitlInterface(audit_log_path) if event_queue is None else None
        logging.info("Web HITL interface initialized (queues wired: %s)", event_queue is not None)

    def _push(self, event: Dict[str, Any]):
        if self.event_queue is not None:
            try:
                self.event_queue.put(event)
            except Exception as e:
                logging.warning(f"Failed to push event to UI: {e}")

    def prompt_user(self, message: str, options: Optional[List[str]] = None,
                   multi_select: bool = False) -> Union[str, List[str]]:
        if self.event_queue is None or self.response_queue is None:
            return self._cli_fallback.prompt_user(message, options, multi_select)

        # HITL_AUTO short-circuit (same convention as CLI)
        if os.getenv("HITL_AUTO") == "1" and options:
            choice = list(options) if multi_select else options[0]
            self._push({
                "type": "hitl_auto",
                "message": message,
                "options": options,
                "auto_choice": choice
            })
            return choice

        prompt_id = f"prompt_{datetime.now().strftime('%H%M%S_%f')}"
        self._push({
            "type": "hitl_prompt",
            "prompt_id": prompt_id,
            "message": message,
            "options": options,
            "multi_select": multi_select
        })

        # Block until UI pushes a response with matching prompt_id
        while True:
            try:
                resp = self.response_queue.get(timeout=600)  # 10 min hard cap
            except Exception:
                logging.warning(f"HITL prompt timed out waiting for UI: {message}")
                # Fall back to first option to keep workflow alive
                return options[0] if options else ""
            if resp.get("prompt_id") == prompt_id:
                return resp.get("value")
            # Other responses (out of order) get re-queued
            self.response_queue.put(resp)

    def show_info(self, message: str, data: Optional[Any] = None):
        if self.event_queue is None:
            return self._cli_fallback.show_info(message, data)
        payload = {"type": "hitl_info", "message": message}
        if data is not None:
            try:
                payload["data"] = data if isinstance(data, (dict, list, str, int, float, bool)) else str(data)[:1000]
            except Exception:
                payload["data"] = str(data)[:1000]
        self._push(payload)

    def show_warning(self, message: str, data: Optional[Any] = None):
        if self.event_queue is None:
            return super().show_warning(message, data)
        self._push({"type": "hitl_warning", "message": message})


def get_hitl_interface(interface_type: str = "cli", **kwargs) -> HitlInterface:
    """
    Factory function to get HITL interface instance.
    Args:
        interface_type: Type of interface ("cli" or "web")
        **kwargs: Additional arguments for interface initialization (event_queue, response_queue for web)
    Returns:
        HitlInterface instance
    """
    if interface_type.lower() == "cli":
        return CliHitlInterface(**kwargs)
    elif interface_type.lower() == "web":
        return WebHitlInterface(**kwargs)
    else:
        raise ValueError(f"Unknown HITL interface type: {interface_type}")


if __name__ == "__main__":
    # Test the HITL interface
    hitl = get_hitl_interface("cli")
    
    # Test single selection
    choice = hitl.prompt_with_audit(
        "Choose a problem type:",
        options=["classification", "regression", "anomaly_detection"],
        context={"test": "single_selection"}
    )
    print(f"User chose: {choice}")
    
    # Test multi-selection
    choices = hitl.prompt_with_audit(
        "Select features to use:",
        options=["feature1", "feature2", "feature3", "feature4"],
        multi_select=True,
        context={"test": "multi_selection"}
    )
    print(f"User chose: {choices}")
    
    # Test info display
    hitl.show_info_with_audit("Test information", {"key": "value"}, {"test": "info_display"})
