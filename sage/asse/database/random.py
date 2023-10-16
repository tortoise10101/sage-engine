
from datetime import datetime
import random
from typing import Optional, Tuple

from sage.asse.database.approx import DatabaseConnectorWithApproxSearch
from sage.asse.corpus import Corpus
from sage.asse.database.iterator import ListIterator
from sage.database.db_connector import DatabaseConnector
from sage.database.db_iterator import DBIterator

class RandomSearchConnector(DatabaseConnectorWithApproxSearch):
    def __init__(self, store: str, exact: DatabaseConnector):
        super(RandomSearchConnector, self).__init__(exact)
        self._store_dir = store
        self.exact = exact
        self.corpus = Corpus(self._store_dir)

    def open(self):
        super(RandomSearchConnector, self).open()
    
    def close(self):
        super(RandomSearchConnector, self).close()
        del self.corpus

    def approximate_search(self, subject: str, predicate: str, obj: str, last_read: Optional[str] = None, as_of: Optional[datetime] = None) -> Tuple[DBIterator, int]:
        _, card = self.exact.search(subject, predicate, obj, last_read, as_of)
        # sample entities index by randint
        ent_ids = random.sample(range(len(self.corpus)), card)
        entities = [self.corpus[i]['uri'] for i in ent_ids]

        subject = subject if (subject is not None) and (not subject.startswith('?')) else ""
        predicate = predicate if (predicate is not None) and (not predicate.startswith('?')) else ""
        obj = obj if (obj is not None) and (not obj.startswith('?')) else ""
        offset = 0 if last_read is None or last_read == '' else int(float(last_read))
        pattern = {'subject': subject, 'predicate': predicate, 'object': obj}
        return ListIterator(entities, pattern, start_offset=offset), card
        
    def from_config(config: dict, exact: DatabaseConnector):

        return RandomSearchConnector(config['store'], exact=exact)
