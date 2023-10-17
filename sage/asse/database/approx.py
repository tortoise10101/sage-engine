from abc import abstractmethod
from datetime import datetime
from typing import Optional, Tuple

from sage.database.db_iterator import DBIterator
from sage.database.db_connector import DatabaseConnector
from sage.asse.database.iterator import ConcatenatedIterator


class DatabaseConnectorWithApproxSearch(DatabaseConnector):
    """A DatabaseConnector subclass that extends functionality to support Approximate Search."""

    def __init__(self, exact: DatabaseConnector):
        super(DatabaseConnectorWithApproxSearch, self).__init__()
        self.exact = exact

    def open(self):
        """Open the database connection"""
        self.exact.open()

    def close(self):
        """Close the database connection"""
        self.exact.close()

    def __enter__(self):
        """Implementation of the __enter__ method from the context manager spec"""
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        """Implementation of the __close__ method from the context manager spec"""
        self.close()

    def __del__(self):
        """Destructor"""
        self.close()

    def search(self, subject: str, predicate: str, obj: str, last_read: str | None = None, as_of: datetime | None = None) -> Tuple[DBIterator, int]:
        if not predicate.startswith('?'):
            if subject.startswith('?') and not obj.startswith('?'):
                return self.approximate_search(subject, predicate, obj, last_read, as_of)
            elif not subject.startswith('?') and obj.startswith('?'):
                return self.approximate_search(subject, predicate, obj, last_read, as_of)
        else:
            if subject.startswith('?') ^ obj.startswith('?'): # XOR
                iter, card = self.exact.search(subject, predicate, obj, last_read, as_of)
                predicates = set()
                while iter.has_next():
                    _, p, _ = iter.next()
                    predicates.add(p)
                res = [self.approximate_search(subject, p, obj, last_read, as_of) for p in predicates]
                card = sum([c for _, c in res])

                return ConcatenatedIterator([i for i, _ in res]), card
        
    @abstractmethod
    def approximate_search(self, subject: str, predicate: str, obj: str, last_read: Optional[str] = None, as_of: Optional[datetime] = None) -> Tuple[DBIterator, int]:
        pass

    @abstractmethod
    def from_config(config: dict, exact: DatabaseConnector):
        """Build a DatabaseConnector from a dictionnary"""
        pass

    def insert(self, subject: str, predicate: str, obj: str) -> None:
        self.exact.insert(subject, predicate, obj)

    def delete(self, subject: str, predicate: str, obj: str) -> None:
        self.exact.delete(subject, predicate, obj)

    def start_transaction(self) -> None:
        """Start a transaction (if supported by this type of connector)"""
        self.exact.start_transaction()

    def commit_transaction(self) -> None:
        """Commit any ongoing transaction (if supported by this type of connector)"""
        self.exact.commit_transaction()

    def abort_transaction(self) -> None:
        """Abort any ongoing transaction (if supported by this type of connector)"""
        self.exact.abort_transaction()

    @property
    def nb_triples(self) -> int:
        """Get the number of RDF triples in the database"""
        return self.exact.nb_triples

    @property
    def nb_subjects(self) -> int:
        """Get the number of subjects in the database"""
        return self.exact.nb_subjects

    @property
    def nb_predicates(self) -> int:
        """Get the number of predicates in the database"""
        return self.exact.nb_predicates

    @property
    def nb_objects(self) -> int:
        """Get the number of objects in the database"""
        return self.exact.nb_objects
