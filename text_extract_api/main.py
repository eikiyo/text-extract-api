# text_extract_api/main.py - Clean version without Railway limitations

import os
import pathlib
import sys
import time
from typing import Optional

import ollama
import redis
from celery.result import AsyncResult
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from pydantic import BaseModel, Field, field_validator

from text_extract_api.celery_app import app as celery_app
from text_extract_api.extract.strategies.strategy import Strategy
from text_extract_api.extract.tasks import ocr_task
from text_extract_api.files.file_formats.file_format import FileFormat, FileField
from text_extract_api.files.storage_manager import StorageManager

# Define base path as text_extract_api - required for keeping absolute namespaces
sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

def storage_profile_exists(profile_name: str) -> bool:
    profile_path = os.path.abspath(
        os.path.join(os.getenv('STORAGE_PROFILE_PATH', './storage_profiles'), f'{profile_name}.yaml'))
    if not os.path.isfile(profile_path) and profile_path.startswith('..'):
        # backward compatibility for ../storage_manager in .env
        sub_profile_path = os.path.normpath(os.path.join('.', profile_path))
        return os.path.isfile(sub_profile_path)
    return True

app = FastAPI(
    title="Text Extract API",
    description="Convert any image, PDF or Office document to Markdown text or JSON structured document with super-high accuracy",
    version="0.2.0"
)

# Connect to Redis
redis_url = os.getenv('REDIS_CACHE_URL', 'redis://redis:6379/1')
redis_client = redis.StrictRedis.from_url(redis_url)

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "text-extract-api",
        "version": "0.2.0",
        "gpu_available": os.getenv('CUDA_VISIBLE_DEVICES') is not None
    }

@app.get("/health")
async def detailed_health():
    """Detailed health check with service status"""
    try:
        # Check Redis
        redis_client.ping()
        redis_status = "healthy"
    except:
        redis_status = "unhealthy"
    
    try:
        # Check Ollama
        ollama.list()
        ollama_status = "healthy"
    except:
        ollama_status = "unhealthy"
    
    return {
        "status": "healthy" if redis_status == "healthy" and ollama_status == "healthy" else "degraded",
        "services": {
            "redis": redis_status,
            "ollama": ollama_status
        },
        "gpu_available": os.getenv('CUDA_VISIBLE_DEVICES') is not None
    }

# ... rest of your endpoints remain the same but without Railway checks ...

@app.post("/ocr")
async def ocr_endpoint(
        strategy: str = Form(...),
        prompt: str = Form(None),
        model: str = Form(...),
        file: UploadFile = File(...),
        ocr_cache: bool = Form(...),
        storage_profile: str = Form('default'),
        storage_filename: str = Form(None),
        language: str = Form('en')
):
    """
    Endpoint to extract text from an uploaded PDF, Image or Office file using different OCR strategies.
    Supports both synchronous and asynchronous processing.
    """
    # Validate input
    try:
        OcrFormRequest(strategy=strategy, prompt=prompt, model=model, ocr_cache=ocr_cache,
                       storage_profile=storage_profile, storage_filename=storage_filename, language=language)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    filename = storage_filename if storage_filename else file.filename
    file_binary = await file.read()
    file_format = FileFormat.from_binary(file_binary, filename, file.content_type)

    print(
        f"Processing Document {file_format.filename} with strategy: {strategy}, ocr_cache: {ocr_cache}, model: {model}, storage_profile: {storage_profile}, storage_filename: {storage_filename}, language: {language}, will be saved as: {filename}")

    # Asynchronous processing using Celery
    task = ocr_task.apply_async(
        args=[file_format.binary, strategy, file_format.filename, file_format.hash, ocr_cache, prompt, model, language,
              storage_profile,
              storage_filename])
    return {"task_id": task.id}

# Include all your other endpoints here without Railway modifications...
