from typing import Dict, List
from sage.database.db_iterator import DBIterator


class ListIterator(DBIterator):
    def __init__(self, list: List, pattern: Dict[str, str], start_offset=0):
        super(ListIterator, self).__init__(pattern)
        self._list = list
        self._index = start_offset
        self._pattern = pattern
        print("Pattern: ", pattern)
        if pattern['object'] == '':
            self._locate = lambda e: (pattern['subject'], pattern['predicate'], e) 
        elif pattern['subject'] == '':
            self._locate = lambda e: (e, pattern['predicate'], pattern['object'])
    
    def has_next(self) -> bool:
        return self._index < len(self._list)
    
    def next(self):
        if not self.has_next():
            raise StopIteration()
        self._index += 1
        
        return self._locate(self._list[self._index - 1])
    
    def last_read(self) -> str:
        return str(self._index - 1)


class ConcatenatedIterator(DBIterator):
    def __init__(self, iterators: List[DBIterator]):
        self._iterators = iterators
        self._current = 0
    
    def has_next(self) -> bool:
        while self._current < len(self._iterators):
            if self._iterators[self._current].has_next():
                return True
            self._current += 1
        return False
    
    def next(self):
        if not self.has_next():
            raise StopIteration()
        return self._iterators[self._current].next()
    
    def last_read(self) -> str:
        return self._iterators[self._current].last_read()

