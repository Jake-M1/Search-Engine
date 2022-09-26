class Posting:
    def __init__(self, docid: int, frequency: int):
        self._docid = docid
        self._frequency = frequency

    #postings are equal when the docid and frequency are the same
    def __eq__(self, other) -> bool:
        if self._docid == other._docid and self._frequency == other._frequency:
            return True
        return False

    def get_docid(self) -> int:
        return self._docid

    def get_frequency(self) -> int:
        return self._frequency

    #string representation of a Posting
    def write(self) -> str:
        return f'({self._docid},{self._frequency})'
