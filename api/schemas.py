from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    query: str
    classification: str
    answer: str