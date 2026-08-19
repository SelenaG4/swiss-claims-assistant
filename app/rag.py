"""Multilingual policy-clause retrieval: an adjuster query in any of DE/FR/EN
should find the right clause regardless of which language it was written in.

How the cross-lingual match actually happens: TF-IDF on its own can't do it
(see app/translate.py's docstring -- "Selbstbehalt" and "franchise" share no
characters). So both the corpus and the query are expanded through
translate_to_all() into all three languages before indexing/matching, and a
single TF-IDF vectorizer runs over that expanded text. A French query for
"franchise" ends up carrying the glossary-substituted German term
"selbstbehalt" too, which is what lets it match a German-only clause.

This means retrieval quality is bounded by the glossary's coverage, not by
TF-IDF -- see the translate.py docstring for that trade-off, stated plainly
rather than glossed over.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.translate import translate_to_all

# A small multilingual stopword list. Without this, generic function words
# ("la", "de", "en", "der", "die", "the", "is") dominate the TF-IDF score for
# any query written in the same language as a clause's original text,
# drowning out the glossary-substituted content words that make cross-lingual
# matching work in the first place.
_STOPWORDS = {
    # French
    "le", "la", "les", "un", "une", "des", "de", "du", "en", "et", "est",
    "à", "dans", "pour", "sur", "avec", "ce", "cette", "ces", "qui", "que",
    "quel", "quelle", "quels", "quelles", "au", "aux", "se", "sa", "son",
    "ses", "ou", "par", "il", "elle", "cas",
    # German
    "der", "die", "das", "und", "ist", "für", "mit", "bei", "von", "nicht",
    "ein", "eine", "einer", "im", "in", "auf", "sich", "sind", "als", "zu",
    "dem", "den", "des", "wird", "werden", "auch", "oder", "wie",
    # English
    "the", "a", "an", "and", "is", "of", "to", "in", "for", "on", "with",
    "this", "that", "are", "as", "by", "or", "be", "was", "were", "at",
}


@dataclass
class PolicyClause:
    id: str
    language: str  # the language it was originally authored in
    topic: str
    text: str
    # Hand-authored reference translations into the other portfolio
    # languages, keyed by language code -- NOT machine-translated. This is a
    # closed set of 9 clauses, small enough that pre-authoring accurate
    # translations was practical, unlike arbitrary user-submitted text (which
    # is why the glossary-substitution mechanism in translate.py still exists
    # and is still what powers cross-lingual retrieval matching -- it has to
    # handle queries this dataset never anticipated). See app/main.py's
    # /policy-query for how the two are combined.
    translations: dict[str, str] = field(default_factory=dict)


class MultilingualPolicyIndex:
    def __init__(self, clauses: list[PolicyClause]):
        self.clauses = clauses
        self._expanded_docs = [self._expand(c.text) for c in clauses]
        self._vectorizer = TfidfVectorizer(stop_words=list(_STOPWORDS))
        self._matrix = self._vectorizer.fit_transform(self._expanded_docs) if clauses else None

    @staticmethod
    def _expand(text: str) -> str:
        """Concatenate the original text with its translation into every
        portfolio language, so shared glossary vocabulary lines up regardless
        of which language the query or the source clause was written in."""
        translations = translate_to_all(text)
        return " ".join([text, *translations.values()])

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.clauses:
            return []
        expanded_query = self._expand(query)
        query_vec = self._vectorizer.transform([expanded_query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(range(len(self.clauses)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "clause_id": self.clauses[i].id,
                "topic": self.clauses[i].topic,
                "source_language": self.clauses[i].language,
                "text": self.clauses[i].text,
                "score": float(scores[i]),
                "translations": self.clauses[i].translations,
            }
            for i in ranked
            if scores[i] > 0
        ]
