from __future__ import annotations

from typing import Any, Dict, Type, Union

from rsl_rl.runners import OnPolicyRunner


class RunnerRegistry:
    """Registry for runner classes to enable dynamic runner instantiation."""

    def __init__(self) -> None:
        self.runner_classes: Dict[str, Type] = {}
    
    def register(self, name: str, runner_class: Type) -> None:
        """Register a runner class with a given name.

        Args:
            name: The name to register the runner under.
            runner_class: The runner class to register.
        """
        self.runner_classes[name] = runner_class
    
    def get_runner_class(self, name: str) -> Type:
        """Get a registered runner class by name.

        Args:
            name: The name of the runner class to retrieve.

        Returns:
            The runner class registered under the given name.

        Raises:
            KeyError: If no runner is registered under the given name.
        """
        return self.runner_classes[name]


runner_registry: RunnerRegistry = RunnerRegistry()