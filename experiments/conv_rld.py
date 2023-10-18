from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()
    dict = {}
    with open(args.text, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split('\t')
            key, value = line[0], line[1].split(',')[0]
            dict[key] = value.replace(' ', '_')
    
    with open(args.file, 'r') as f, open(args.output, 'w') as out:
        for line in f:
            line = line.strip()
            s, p, o = line.split('\t')
            s = dict[s]
            o = dict[o]
            s = 'http://example.org/' + s
            p = 'http://example.org/' + p
            o = 'http://example.org/' + o
            out.write(f'<{s}> <{p}> <{o}> .\n')
