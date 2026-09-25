from torch import nn


class Registry:
    def __init__(self):
        self._cls_dict = {}

    def add(self, cls: nn.Module) -> nn.Module:
        self._cls_dict[cls.__name__.lower()] = cls
        return cls

    def get(self, cls_name: str) -> nn.Module:
        if cls_name.lower() not in self._cls_dict:
            raise KeyError(f'"{cls_name}" is not a name of a registered class')
        return self._cls_dict[cls_name.lower()]

    def __repr__(self) -> str:
        return f"Registry({self._cls_dict})"
