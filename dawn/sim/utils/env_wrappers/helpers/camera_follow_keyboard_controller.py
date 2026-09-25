from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CameraFollowKeyBindings:
    toggle_follow: str = "G"

    orbit_left: str = "LEFT"
    orbit_right: str = "RIGHT"

    orbit_up: str = "UP"
    orbit_down: str = "DOWN"

    zoom_in: str = "PAGE_UP"
    zoom_out: str = "PAGE_DOWN"


class CameraFollowKeyboardController:
    def __init__(self):
        self._keys = CameraFollowKeyBindings()
        self._follow_enabled = True
        self._pressed_keys: set[str] = set()

        self._input = None
        self._keyboard = None
        self._subscription = None
        self._event_type = None

    @property
    def follow_enabled(self) -> bool:
        return self._follow_enabled

    @property
    def horizontal_orbit_direction(self) -> int:
        return self._pressed_direction(
            positive_key=self._keys.orbit_right,
            negative_key=self._keys.orbit_left,
        )

    @property
    def vertical_orbit_direction(self) -> int:
        return self._pressed_direction(
            positive_key=self._keys.orbit_up,
            negative_key=self._keys.orbit_down,
        )

    @property
    def zoom_direction(self) -> int:
        return self._pressed_direction(
            positive_key=self._keys.zoom_out,
            negative_key=self._keys.zoom_in,
        )

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
        self._pressed_keys.clear()

    def _on_keyboard_event(self, event: Any, *args: object) -> bool:
        if event.type == self._event_type.KEY_PRESS:
            self.press_key(event.input.name)
        elif event.type == self._event_type.KEY_RELEASE:
            self.release_key(event.input.name)
        return True

    def press_key(self, key: str) -> None:
        if key == self._keys.toggle_follow:
            self._follow_enabled = not self._follow_enabled
            print(f"CameraFollowWrapper enabled: {self._follow_enabled}")
            return

        self._pressed_keys.add(key)

    def release_key(self, key: str) -> None:
        self._pressed_keys.discard(key)

    def _pressed_direction(self, positive_key: str, negative_key: str) -> int:
        direction = 0
        if positive_key in self._pressed_keys:
            direction += 1
        if negative_key in self._pressed_keys:
            direction -= 1
        return direction
