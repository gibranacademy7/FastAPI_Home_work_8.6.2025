import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")

def get_phrases(text, phrases):
    matcher = PhraseMatcher(nlp.vocab)
    patterns = [nlp.make_doc(phrase) for phrase in phrases]
    matcher.add("CUSTOM_PHRASES", patterns)

    doc = nlp(text)
    matches = matcher(doc)
    return [doc[start:end].text for match_id, start, end in matches]
