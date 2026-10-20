TRAINER_REGISTRY = {}


def register_trainer(name):
    def decorator(cls):
        TRAINER_REGISTRY[name] = cls
        return cls
    return decorator


def get_trainer(name: str):
    return TRAINER_REGISTRY.get(name)
