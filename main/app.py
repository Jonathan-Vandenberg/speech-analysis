import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes_scripted import router as scripted_router
from .routes_unscripted import router as unscripted_router


FRONTEND_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001").split(",")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()  # Back to INFO now that Panphon is fixed


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scripted_router, prefix="/analyze")
app.include_router(unscripted_router, prefix="/analyze")


logger = logging.getLogger("speech_analyzer")
if not logger.handlers:
    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

 
