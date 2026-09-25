from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnvIndexKeyBindings:
    previous_env: str = "LEFT_BRACKET"
    next_env: str = "RIGHT_BRACKET"


class EnvIndexKeyboardController:
    def __init__(self, num_envs: int):
        self._num_envs = num_envs
        self._env_idx = 0
        self._keys = EnvIndexKeyBindings()

        self._input = None
        self._keyboard = None
        self._subscription = None
        self._event_type = None

    @property
    def env_idx(self) -> int:
        return self._env_idx

    def setup(self) -> None:
        if self._subscription is not None:
            return

        import carb  # type: ignore
        import omni  # type: ignore

        app_window = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = app_window.get_keyboard()
        self._event_type = carb.input.KeyboardEventType
        self._subscription = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

    def close(self) -> None:
        if self._subscription is None:
            return

        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
        self._subscription = None

    def _on_keyboard_event(self, event: Any, *args: object) -> bool:
        if event.type == self._event_type.KEY_PRESS:
            self.press_key(event.input.name)
        return True

    def press_key(self, key: str) -> None:
        if key == self._keys.previous_env:
            self._move_env_index(-1)
        elif key == self._keys.next_env:
            self._move_env_index(1)

    def _move_env_index(self, offset: int) -> None:
        next_env_idx = (self._env_idx + offset) % self._num_envs
        if next_env_idx == self._env_idx:
            return

        self._env_idx = next_env_idx
        print(f"Visualization env_idx: {self._env_idx}")
