from fastapi import FastAPI, HTTPException
from typing import Dict
from services.task_extraction import TaskExtractionService
from services.email_classification import EmailClassificationService
from schemas.task import TaskExtractionRequest, TaskExtractionResponse
from schemas.email_classification import TrainingResponse, PredictionRequest, PredictionResponse

app = FastAPI(
    title="Declutter API",
    description="Backend API for the Declutter application",
    version="1.0.0"
)

# Initialize the email classification service
email_classifier = EmailClassificationService()

@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Welcome to Declutter API"}

@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}

@app.post("/api/extract-tasks", response_model=TaskExtractionResponse)
async def extract_tasks(request: TaskExtractionRequest) -> TaskExtractionResponse:
    """
    Extract actionable tasks from user context.
    """
    try:
        tasks = await TaskExtractionService.extract_tasks(request.context)
        return TaskExtractionResponse(tasks=tasks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train-email-classifier", response_model=TrainingResponse)
async def train_email_classifier() -> TrainingResponse:
    """
    Train the BERT model for email classification using the provided dataset.
    """
    try:
        dataset_path = "assets/datasets/email_pillars_dataset_v1.csv"
        result = await email_classifier.train_model(dataset_path)
        return TrainingResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/classify-email", response_model=PredictionResponse)
async def classify_email(request: PredictionRequest) -> PredictionResponse:
    """
    Classify the given email text into categories (career, finance, physical, mental).
    """
    try:
        result = await email_classifier.predict(request.text)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
