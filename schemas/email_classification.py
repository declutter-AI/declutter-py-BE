from pydantic import BaseModel
from typing import Dict, Union

class TrainingResponse(BaseModel):
    accuracy: float
    num_classes: int
    epochs_trained: int

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    category: str
    confidence: float
