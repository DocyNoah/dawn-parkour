# Isaac Sim can bundle typing_extensions without the sentinel required by tyro.
from dawn.utils.tyro_compat import ensure_tyro_runtime_compat as _ensure_tyro_runtime_compat

_ensure_tyro_runtime_compat()
