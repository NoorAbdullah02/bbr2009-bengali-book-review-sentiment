from huggingface_hub import snapshot_download as s
s('google-bert/bert-base-multilingual-cased', local_dir='./models/mbert', allow_patterns=['*.json', '*.txt', 'model.safetensors'])
print('ok mbert', flush=True)
