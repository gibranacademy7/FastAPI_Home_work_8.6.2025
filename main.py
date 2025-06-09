from fastapi import FastAPI
from markdown_it.common.entities import entities
from pydantic import BaseModel
from spacy.lang.el.tokenizer_exceptions import token
from spacy.matcher import PhraseMatcher
from starlette.responses import HTMLResponse
import bl
import spacy

nlp = spacy.load("en_core_web_sm")

app = FastAPI(title="Example API", description="A simple FastAPI with Swagger UI", version="1.0")

@app.get("/welcome")
def read_root():
    return {"message": "Welcome to the FastAPI example!"}


#====================  1  ==========================


@app.get("/get-text-and-label/text/")
def get_text_and_label(text: str):
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return entities


#====================  2 ==========================


@app.get("/get-person-in-text/text")
def get_person_ent(text:str):
    doc = nlp(text)
    entities = [{"Person":ent.text} for ent in doc.ents if ent.label_ == "PERSON"]
    return entities

#====================  3  ==========================


@app.get("/get-lemma-in-text/text")
def get_lemma(text:str):
    doc = nlp(text)
    lemmas = [{f"{token.text:7} ->": token.lemma_} for token in doc ]
    return lemmas

#====================  4 ==========================


@app.get("/not-stop-words/text")
def get_not_stop_words(text:str):
    doc = nlp(text)
    not_stop_words = [{token.text} for token in doc if not token.is_stop]
    return f"Not Stop Words in Text: {not_stop_words}"

#====================  5 ==========================


@app.get("/get-stop-words/text")
def get_stop_words(text:str):
    nlp.vocab["powerful"].is_stop = True
    doc = nlp(text)

    stop_words = [{token.text} for token in doc if token.is_stop]
    return f"stop_words are: {stop_words}"

#====================  6 ==========================

from spacy.matcher import PhraseMatcher

@app.get("/phrasematcher/text")
def get_phrases(text: str):
    matcher = PhraseMatcher(nlp.vocab)
    phrases = ["artificial intelligence","Artificial Intelligence"]
    patterns = [nlp(p) for p in phrases]

    matcher.add("AI_PHRASE", patterns)

    doc = nlp(text)
    matches = matcher(doc)

    results = []
    for match_id, start, end in matches:
        results.append(doc[start:end].text)

    return {"matches": results}

#====================  7 ==========================

@app.get("/get-token-details/text")
def get_token_details(text:str):
    doc = nlp(text)

    details = [{"token": token.text, "token.pos": token.pos_, "description": spacy.explain(token.pos_)} for token in doc]
    return details

#====================  8 ==========================

from spacy.language import Language

@Language.component("custom_separator")
def custom_separator(doc):
    for token in doc[:-1]:
        if token.text == '^':
            doc[token.i + 1].is_sent_start = True
    return doc

nlp.add_pipe("custom_separator", before="parser")

@app.get("/custom-sentence-split/text")
def split_sentences(text: str):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return {"sentences": sentences}

#======================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)