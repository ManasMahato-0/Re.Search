

import math
import re
from collections import Counter

# ---------------------------------------------------------
# TOKENIZER
# ---------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset("""
a an and are as at be by for from has have in is it its of on or that the
this to was were will with what which how when where who why
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


# ---------------------------------------------------------
# BM25 (Okapi BM25)
# ---------------------------------------------------------
class BM25:
    

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)

        # Per-document word frequency tables and lengths
        self.doc_freqs: list[Counter] = []
        self.doc_lengths: list[int] = []
        total_len = 0

        for doc in corpus:
            tokens = tokenize(doc)
            self.doc_freqs.append(Counter(tokens))
            self.doc_lengths.append(len(tokens))
            total_len += len(tokens)

        # Average document length 
        self.avgdl = total_len / self.corpus_size if self.corpus_size > 0 else 1.0

        df: dict[str, int] = {}
        for freq in self.doc_freqs:
            for word in freq.keys():
                df[word] = df.get(word, 0) + 1

        # Inverse document frequency 
        self.idf: dict[str, float] = {}
        N = self.corpus_size
        for word, n_q in df.items():
            self.idf[word] = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1)

    def get_scores(self, query_string: str) -> list[float]:
        """Return one BM25 score per document for this query."""
        query_words = tokenize(query_string)
        scores = []
        for i, doc_freq in enumerate(self.doc_freqs):
            doc_score = 0.0
            doc_length = self.doc_lengths[i]
            for word in query_words:
                if word in self.idf:
                    word_freq = doc_freq.get(word, 0)
                    word_idf = self.idf[word]

                    # The BM25 saturation formula
                    numerator = word_freq * (self.k1 + 1)
                    denominator = word_freq + self.k1 * (
                        1 - self.b + self.b * (doc_length / self.avgdl)
                    )
                    doc_score += word_idf * (numerator / denominator)
            scores.append(doc_score)
        return scores
