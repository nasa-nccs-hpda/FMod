from modulus.registry import ModelRegistry

model_registry = ModelRegistry()
model_registry.list_models()
SFNO_Model = model_registry.factory("SFNO")
print( SFNO_Model.__class__.__name__ )