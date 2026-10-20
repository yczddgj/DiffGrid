import importlib

EXTRACTOR_REGISTRY = {}
_MODULES = {'resnet18': 'resnet', 'resnet50': 'resnet', 'clip': 'clip',
            'dino': 'dino', 'dreamsim': 'dreamsim'}


def register_extractor(name):
    def decorator(cls):
        EXTRACTOR_REGISTRY[name] = cls
        return cls
    return decorator


def get_extractor(name: str):
    if name not in EXTRACTOR_REGISTRY and name in _MODULES:
        importlib.import_module(f'{__package__}.{_MODULES[name]}')
    return EXTRACTOR_REGISTRY.get(name)


def list_extractors():
    """Return a list of available extractor method names."""
    return sorted(set(_MODULES) | set(EXTRACTOR_REGISTRY))
