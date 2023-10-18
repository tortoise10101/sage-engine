import requests
from argparse import ArgumentParser
from yaml import load, FullLoader
import ujson as json
import os

data = {
    'query': '',
    'defaultGraph': '',
    'next': None,
    'is_asse': True,
}
def execute_queries(queries, data):
    output = []
    for query in queries:
        data['query'] = query
        response = requests.post(args.url, json=data)
        if response.status_code != 200:
            print(response.text)
            raise
        obj = json.loads(response.text)
        obj['query'] = query
        output.append(json.dumps(obj))
    
    return output

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--url', type=str, default='http://localhost:8000/sparql')
    parser.add_argument('--query_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--config', type=str, default='')
    args = parser.parse_args()

    config = load(open(args.config), Loader=FullLoader)
    data['defaultGraph'] = config['graphs'][0]['uri']
    with open(args.query_file, 'r') as f:
        queries = [l.strip() for l in f.readlines()]
    
    data['is_asse'] = False
    exact_output = execute_queries(queries, data)
    data['is_asse'] = True
    asse_output = execute_queries(queries, data)
        
    statistics = {
        'score': 0,
        'exact total': 0,
        'asse total': 0,
        'exact mean length': 0,
        'asse mean length': 0,
        'number of queries': len(queries),
    }
    for exact, asse in zip(exact_output, asse_output):
        exact, asse = json.loads(exact), json.loads(asse)

        s1 = set([str(s) for s in exact['bindings']])
        s2 = set([str(s) for s in asse['bindings']])
        union = s1.union(s2)
        if len(union) > 0:
            statistics['score'] += len(s1.intersection(s2)) / len(s1.union(s2))
        else:
            statistics['score'] += 1
        statistics['exact total'] += len(s1)
        statistics['asse total'] += len(s2)
        #if len(s1) > 0 or len(s2) > 0:
        #    print(s1, s2, len(s1.intersection(s2)) / len(s1.union(s2)))
    statistics['score'] /= len(exact_output)
    statistics['exact mean length'] = statistics['exact total'] / len(exact_output)
    statistics['asse mean length'] = statistics['asse total'] / len(asse_output)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'exact.json'), 'w') as f:
        f.write('\n'.join(exact_output))
    with open(os.path.join(args.output_dir, 'asse.json'), 'w') as f:
        f.write('\n'.join(asse_output))
    with open(os.path.join(args.output_dir, 'statistics.json'), 'w') as f:
        f.write(json.dumps(statistics))
    
    print(statistics)
