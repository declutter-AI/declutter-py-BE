from pydantic import BaseModel
from typing import List

class TaskExtractionRequest(BaseModel):
    context: str

class TaskExtractionResponse(BaseModel):
    tasks: List[str]
