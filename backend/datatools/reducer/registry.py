REDUCER_REGISTRY = {}


def register_reducer(name):
    def decorator(cls):
        REDUCER_REGISTRY[name] = cls
        return cls
    return decorator


def get_reducer(name: str):
    # print(REDUCER_REGISTRY)
    return REDUCER_REGISTRY.get(name)


def list_reducers():
    """Return a list of available reducer method names."""
    return sorted(REDUCER_REGISTRY.keys())
