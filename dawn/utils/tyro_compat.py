from __future__ import annotations

import typing_extensions


def ensure_tyro_runtime_compat() -> None:
    if hasattr(typing_extensions, "NoExtraItems"):
        return

    class NoExtraItems:
        pass

    typing_extensions.NoExtraItems = NoExtraItems
