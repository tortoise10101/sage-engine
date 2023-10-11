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
import codecs

from sage.cli.parsers import ParserFactory

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

def gzip_str(str):
    return codecs.encode(str.encode('utf-8'), 'zlib')
    # return gzip.compress(str.encode('utf-8'))

def load_entity(file):
    format = file.split('.')[-1]
    parser = ParserFactory.create_parser(format)
    parser.parsefile(file)

    entities = set()
    for h, r, t in parser.readlines():
        entities.add(h)
        entities.add(t)
    
    return entities

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

@click.command()
@click.argument("config")
def build_index(config):
    with open(config) as f:
        config = json.load(f)
    file = config['file']
    entities = load_entity(file)

    ctx_encoder = DPRContextEncoder.from_pretrained(
        config['model']).to(device=device)
    ctx_encoder.eval()
    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(
        config['model'])
    
    output_path = os.path.join(config['store'], f'embed/embed.json.gz.records')
    batch_size = 64
    cur_offset = 0
    offsets = []
    vectors = []
    entities = sorted(list(entities))
    with gzip.open(output_path, 'wb') as fp:
        for i in tqdm(range(0, len(entities), batch_size)):
            ent_batch = entities[i:i+batch_size]
            embeddings = embed_entity(ent_batch, ctx_encoder, ctx_tokenizer)
            cur_offset = write(cur_offset, offsets, fp, ent_batch, embeddings)
            vectors.append(embeddings)
        ent_batch = entities[i:]
        embeddings = embed_entity(ent_batch, ctx_encoder, ctx_tokenizer)
        cur_offset = write(cur_offset, offsets, fp, ent_batch, embeddings)
        vectors.append(embeddings)

    output_file = os.path.join(config['store'], f'embed/vector.npy')
    with open(output_file, 'wb') as f:
        np.save(f, np.array(offsets, dtype=np.int64), allow_pickle=False)
    
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

    vectors = np.concatenate(vectors, axis=0)
    index.train(vectors)
    index.add(vectors)
    faiss.write_index(index, os.path.join(config['store'], 'embed/index.faiss'))

