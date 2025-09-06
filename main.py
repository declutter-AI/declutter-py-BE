from fastapi import FastAPI, HTTPException
from typing import Dict
from services.task_extraction import TaskExtractionService
from schemas.task import TaskExtractionRequest, TaskExtractionResponse
from services.gmail_service import router as gmail_router


app = FastAPI(
    title="Declutter API",
    description="Backend API for the Declutter application",
    version="1.0.0"
)

app.include_router(gmail_router, prefix="/gmail", tags=["gmail"])

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
