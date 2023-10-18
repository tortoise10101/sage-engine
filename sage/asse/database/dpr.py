from datetime import datetime
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import os

from sage.asse.database.approx import DatabaseConnectorWithApproxSearch
from sage.asse.corpus import Corpus
from sage.asse.database.iterator import ListIterator
from sage.database.db_connector import DatabaseConnector
from sage.database.db_iterator import DBIterator

class DensePassageRetrievalConnector(DatabaseConnectorWithApproxSearch):
    def __init__(self, store: str, index_file: str, model_name: str, exact: DatabaseConnector):
        super(DensePassageRetrievalConnector, self).__init__(exact)
        self._store_dir = store
        self.exact = exact
        self.index = faiss.read_index(index_file)
        self.corpus = Corpus(self._store_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def open(self):
        super(DensePassageRetrievalConnector, self).open()
    
    def close(self):
        super(DensePassageRetrievalConnector, self).close()
        del self.corpus

    def cls_pooling(self, model_output, attention_mask):
        return model_output[0][:,0]
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def approximate_search(self, subject: str, predicate: str, obj: str, last_read: Optional[str] = None, as_of: Optional[datetime] = None) -> Tuple[DBIterator, int]:
        _, card = self.exact.search(subject, predicate, obj, last_read, as_of)
        # sample entities index by randint
        inputs = f"{subject} {predicate} {obj}"
        tokens = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**tokens)
        if 'pooler_output' in outputs:
            embeddings = outputs['pooler_output'].detach().cpu().numpy()
        else:
            embeddings = self.mean_pooling(outputs, tokens['attention_mask'])

        D, I = self.index.search(embeddings, card)
        entities = [self.corpus[i]['uri'] for i in I[0]]
        print(inputs)
        print(D.shape, I.shape, entities[:10])

        subject = subject if (subject is not None) and (not subject.startswith('?')) else ""
        predicate = predicate if (predicate is not None) and (not predicate.startswith('?')) else ""
        obj = obj if (obj is not None) and (not obj.startswith('?')) else ""
        offset = 0 if last_read is None or last_read == '' else int(float(last_read))
        pattern = {'subject': subject, 'predicate': predicate, 'object': obj}
        return ListIterator(entities, pattern, start_offset=offset), card
        
    def from_config(config: dict, exact: DatabaseConnector):

        index_file = os.path.join(config['store'], f'embed/index.faiss')
        return DensePassageRetrievalConnector(
            store=config['store'],
            index_file=index_file,
            model_name=config['model'],
            exact=exact)
