import app
import optimizer
import os

print(f"Models Dir: {app.MODELS_DIR}")
models = app.get_available_gguf_models()
print(f"Found {len(models)} models.")
for m in models:
    path = app.MODELS_DIR / m
    meta = optimizer.read_gguf_metadata(str(path))
    if meta:
        print(f"\n==== {m} ====\n{meta.get('tokenizer.chat_template', 'NOT FOUND')}")
    else:
        print(f"\n==== {m} ====\nERROR READING METADATA")
