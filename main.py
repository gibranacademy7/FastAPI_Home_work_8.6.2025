from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import HTMLResponse
from typing import List
import bl

# Create FastAPI instance
app = FastAPI(title="Example API", description="A simple FastAPI with Swagger UI", version="1.0")

# Define a Pydantic model for input data
class Item(BaseModel):
    name: str
    price: float
    in_stock: bool = True

# Define a simple GET endpoint
@app.get("/welcome")
def read_root():
    return {"message": "Welcome to the FastAPI example!"}

# Define a POST endpoint
@app.post("/send-item/items/")
def create_item(item: Item):
    return {"item_received": item}

@app.delete("/delete-item")
def read_root():
    return {"message": "item deleted"}

@app.put("/update-item")
def read_root():
    return {"message": "item updated"}

@app.get("/sum")
def sum_two_get(a: int, b: int):
    result = bl.sum_two(a, b)
    return {"sum": result}

class TwoNumbers(BaseModel):
    a: int
    b: int

@app.post("/sum-post")
def sum_two_post(two_numbers: TwoNumbers):
    result = bl.sum_two(two_numbers.a, two_numbers.b)
    return {"sum": result}

@app.get("/get-entities")
def get_entities(text: str):
    print(f"========= input get_entities: {text}")
    result = bl.get_entities(text)
    return {"entities": result}

@app.get("/get-person")
def get_person(text: str):
    print(f"========= input get_person: {text}")
    result = bl.get_person(text)
    return {"people": result}


@app.get("/phrasematcher")
def get_phrases(text: str, phrases: List[str]):
    result = bl.get_phrases(text, phrases)
    return {"matched_phrases": result}
