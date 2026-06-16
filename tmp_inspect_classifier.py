import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

config = {
    'tokenizer': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'labels': ['Négatif', 'Neutre', 'Positif']
}
model_dir = 'models/sentiment'

candidate = None
for name in os.listdir(model_dir):
    path = os.path.join(model_dir, name)
    if os.path.isdir(path) and os.path.exists(os.path.join(path, 'config.json')):
        candidate = path
        break

model_path = candidate or model_dir
print('using model_path:', model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(config['labels']))
tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
clf = pipeline('text-classification', model=model, tokenizer=tokenizer, device=-1, top_k=None)
text = "J'adore ce produit, il est incroyable"
res = clf(text)
print('type:', type(res))
print('repr:', repr(res))
print('len:', len(res) if hasattr(res, '__len__') else 'no len')
if isinstance(res, list):
    print('first type:', type(res[0]))
    print('first repr:', repr(res[0]))
    if isinstance(res[0], list):
        print('inner len:', len(res[0]))
        print('inner first type:', type(res[0][0]))
        print('inner first repr:', repr(res[0][0]))
