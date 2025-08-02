# backend/models.py
from pydantic import BaseModel
from typing import List




class FeedbackRequest(BaseModel):
    query_id: str
    feedback: str  # "up" or "down"

class RegisterRequest(BaseModel):
    username: str
    email:    str
    password: str

class TokenResponse(BaseModel):
    access_token: str

class UploadPDFResponse(BaseModel):
    document_id: str
    snippet:     str

class QueryRequest(BaseModel):
    document_id: str
    query:       str

class QueryResponse(BaseModel):
    answer: str







class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]