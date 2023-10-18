import ujson as json
import logging
import os
import torch
from typing import List
import numpy as np
import base64
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
)
import click
from tqdm import tqdm
import gzip
import faiss
from yaml import load, FullLoader
from datasets import Dataset
from datasets import Features, Sequence, Value
from functools import partial

from sage.cli.parsers import ParserFactory
from sage.asse.corpus import gzip_str, gunzip_str, Corpus

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_entity_relation(file):
    format = file.split('.')[-1]
    parser = ParserFactory.create_parser(format)
    parser.parsefile(file)

    entities = set()
    relations = set()
    for s, p, o in tqdm(parser):
        entities.add(s)
        relations.add(p)
        if '"' in o:
            continue
        entities.add(o)
    
    entities = sorted(list(entities))
    relations = sorted(list(relations))
    return entities, relations

def embed_entity(ent_batch: List, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast) -> np.ndarray:
    input_ids = ctx_tokenizer(
        ent_batch, truncation=True, padding="longest", return_tensors="pt"
    )["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device=device), return_dict=True).pooler_output

    return embeddings.detach().cpu().to(dtype=torch.float16).numpy()

def write(cur_offset, offsets, fp, ent_batch, embeddings):
    assert len(ent_batch) == embeddings.shape[0]
    assert len(embeddings.shape) == 2
    for i, entity in enumerate(ent_batch):
        obj = {
            'uri': entity,
            'vector': base64.b64encode(embeddings[i].astype(np.float16)).decode('ascii'),
        }
        jstr_gz = gzip_str(json.dumps(obj))
        offsets.append(cur_offset)
        fp.write(jstr_gz)
        cur_offset += len(jstr_gz)

    return cur_offset

def embed(documents: dict, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast) -> dict:
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(
        documents["uri"], truncation=True, padding="longest", return_tensors="pt"
    )["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device=device), return_dict=True).pooler_output

    return {"vector": embeddings.detach().cpu().numpy()}

@click.command()
@click.argument("config")
def build_index(config):
    config = load(open(config), Loader=FullLoader)
    print(config)
    file = config['graphs'][0]['file']
    entities, relations = load_entity_relation(file)
    with open(os.path.join(config['store'], 'relations.txt'), 'w') as f:
        for relation in relations:
            f.write(f'{relation}\n')
    with open(os.path.join(config['store'], 'entities.txt'), 'w') as f:
        for entity in entities:
            f.write(f'{entity}\n')

    file = os.path.join(config['store'], 'entities.txt')
    print('loading entities from', file, len(entities))
    entities = Dataset.from_text(file)
    print('loaded', len(entities), 'entities')
    entities = entities.rename_column('text', 'uri')

    ctx_encoder = DPRContextEncoder.from_pretrained(
        config['model']).to(device=device)
    ctx_encoder.eval()
    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(
        config['model'])
    
    new_features = Features(
        {"uri": Value("string"), "vector": Sequence(Value("float32"))}
    )
    print(entities)
    print(entities['uri'][:10])
    # eval mode
    ctx_encoder.eval()
    entities = entities.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer),
        batched=True,
        batch_size=16,
        features=new_features,
    )

    passages_path = os.path.join(config['store'], 'embed')
    entities.save_to_disk(passages_path)

    print("Building index...")
    index = faiss.IndexHNSWFlat(config['embed_dim'], 128, faiss.METRIC_INNER_PRODUCT)
    entities.add_faiss_index("vector", custom_index=index)

    index_path = os.path.join(config['store'], 'embed/index.faiss')
    entities.get_index("vector").save(index_path)

'''
def build_index(config):
    config = load(open(config), Loader=FullLoader)
    print(config)
    file = config['graphs'][0]['file']
    entities, relations = load_entity_relation(file)

    ctx_encoder = DPRContextEncoder.from_pretrained(
        config['model']).to(device=device)
    ctx_encoder.eval()
    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(
        config['model'])
    
    output_path = os.path.join(config['store'], f'embed/passages_1_of_1.json.gz.records')
    batch_size = 16
    cur_offset = 0
    offsets = []
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    # with gzip.open(output_path, 'wb') as fp:
    with open(output_path, 'wb') as fp:
        for i in tqdm(range(0, len(entities), batch_size)):
            ent_batch = entities[i:i+batch_size]
            embeddings = embed_entity(ent_batch, ctx_encoder, ctx_tokenizer)
            cur_offset = write(cur_offset, offsets, fp, ent_batch, embeddings)
            # torch cache clear
            torch.cuda.empty_cache()
        ent_batch = entities[i:]
        if len(ent_batch) > 0:
            embeddings = embed_entity(ent_batch, ctx_encoder, ctx_tokenizer)
            cur_offset = write(cur_offset, offsets, fp, ent_batch, embeddings)
        offsets.append(cur_offset)

    with open(os.path.join(config['store'], f'embed/offsets_{1}_of_{1}.npy'), 'wb') as f:
        np.save(f, np.array(offsets, dtype=np.int64), allow_pickle=False)
    
    with open(os.path.join(config['store'], 'relations.txt'), 'w') as f:
        for relation in relations:
            f.write(f'{relation}\n')

    if config['scalar_quantizer'] > 0:
        if config['scalar_quantizer'] == 16:
            sq = faiss.ScalarQuantizer.QT_fp16
        elif config['scalar_quantizer'] == 8:
            sq = faiss.ScalarQuantizer.QT_8bit
        elif config['scalar_quantizer'] == 4:
            sq = faiss.ScalarQuantizer.QT_4bit
        elif config['scalar_quantizer'] == 6:
            sq = faiss.ScalarQuantizer.QT_6bit
        else:
            raise ValueError(f'unknown --scalar_quantizer {config["scalar_quantizer"]}')
        # see https://github.com/facebookresearch/faiss/blob/master/benchs/bench_hnsw.py
        index = faiss.IndexHNSWSQ(config['embed_dim'], sq, 128)
    else:
        index = faiss.IndexHNSWFlat(config['embed_dim'], 128, faiss.METRIC_INNER_PRODUCT)


    corpus = Corpus(config['store'])
    vectors = np.array([c['vector'] for c in corpus])
    print(vectors.shape)
    index.train(vectors)
    index.add(vectors)
    faiss.write_index(index, os.path.join(config['store'], 'embed/index.faiss'))
'''