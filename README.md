# Declutter API

A FastAPI-based backend for the Declutter application.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

To run the application:

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## Available Endpoints

- `GET /`: Welcome message
- `GET /health`: Health check endpoint
- `GET /api/dummy`: Dummy API endpoint that returns example data

## API Documentation

Once the server is running, you can access:
- Swagger UI documentation at: `http://localhost:8000/docs`
- ReDoc documentation at: `http://localhost:8000/redoc`