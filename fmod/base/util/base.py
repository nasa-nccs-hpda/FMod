

class FMSingleton(object):
    _instance = None
    _instantiated = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            inst = cls()
            cls._instance = inst
            cls._instantiated = cls
        return cls._instance