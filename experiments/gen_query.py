from argparse import ArgumentParser
import os
import random
import requests
from yaml import load, FullLoader
import ujson as json

from sage.asse.corpus import Corpus

data = {
    'query': '',
    'defaultGraph': '',
    'next': None,
    'is_asse': False,
}
def check_length(query, data):
    data['query'] = query
    response = requests.post(args.url, json=data)
    if response.status_code != 200:
        print(response.text)
        raise
    obj = json.loads(response.text)

    return len(obj['bindings'])

query_templates = [
    'select * where { ?s <__PREDICATE__> <__OBJECT__> . }',
    'select * where { <__SUBJECT__> <__PREDICATE__> ?o . }',
    'select * where { <__SUBJECT__> ?p ?o . }',
    'select * where { <__SUBJECT__> <__PREDICATE__> ?o1 . <__SUBJECT__> <__PREDICATE__> ?o2 . }',
    'select * where { <__SUBJECT__> <__PREDICATE__> ?o1 . ?o1 <__PREDICATE__> ?o2 . }',
    'select * where { <__SUBJECT__> ?p ?o . ?o ?p2 <__OBJECT__> . }',
]
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--store', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str)
    parser.add_argument('--url', type=str, default='http://localhost:8000/sparql')

    args = parser.parse_args()
    config = load(open(args.config), Loader=FullLoader)
    random.seed(args.seed)
    data['defaultGraph'] = config['graphs'][0]['uri']

    corpus = Corpus(args.store)
    with open(os.path.join(args.store, 'relations.txt'), 'r') as f:
        relations = set([l.strip() for l in f.readlines()])
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    for i in range(len(query_templates)):
        output_file = os.path.join(args.output_dir, f'query{i}.txt')
        with open(output_file, 'w') as f:
            cnt = 0
            while cnt < args.n:
                templates = query_templates[i]
                while '<__SUBJECT__>' in templates:
                    templates = templates.replace('<__SUBJECT__>', corpus[random.randint(0, len(corpus)-1)]['uri'], 1)
                while '<__PREDICATE__>' in templates:
                    templates = templates.replace('<__PREDICATE__>', random.choice(list(relations)), 1)
                while '<__OBJECT__>' in templates:
                    templates = templates.replace('<__OBJECT__>', corpus[random.randint(0, len(corpus)-1)]['uri'], 1)

                _len = check_length(templates, data)
                if _len > 0:
                    f.write(templates + '\n')
                    cnt += 1
