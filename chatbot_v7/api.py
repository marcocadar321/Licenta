"""
api.py - Server FastAPI.

Pornire:
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import persistence
import security
from nlp import Chatbot



bot    = Chatbot()
bearer = HTTPBearer(auto_error=False)

app = FastAPI(
    title       = "ChatBot NLP API",
    description = "Chatbot semantic cu autentificare JWT, rate limiting si NLP local.",
    version     = "7.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = config.ALLOWED_ORIGINS,
    allow_methods     = ["GET", "POST"],
    allow_headers     = ["Authorization", "Content-Type"],
    allow_credentials = True,
)

if config.STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")




class AuthRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

class ChatResponse(BaseModel):
    response: str
    matched:  bool
    score:    float | None = None
    mode:     str = "nlp"

class StatsResponse(BaseModel):
    total:          int
    matched:        int
    match_rate_pct: int
    avg_time_ms:    float
    nlp_mode:       str = "unknown"




def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
) -> str:
    if not credentials:
        raise HTTPException(status_code=401, detail="Autentificare necesara.")
    username = security.decode_token(credentials.credentials)
    if not username:
        raise HTTPException(status_code=401, detail="Token invalid sau expirat.")
    return username


def enforce_rate_limit(key: str, limit: int) -> None:
    allowed, retry_after = security.check_rate_limit(key, limit)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Prea multe cereri. Incearca dupa {retry_after}s.",
        )




@app.get("/", include_in_schema=False)
def root():
    ui = config.STATIC_DIR / "index.html"
    if ui.exists():
        return FileResponse(str(ui))
    return {"message": "ChatBot NLP API v7.0"}


@app.get("/health")
def health():
    return {
        "status":   "ok",
        "patterns": len(bot.kb.patterns),
        "nlp_mode": bot._nlp_mode,
    }


@app.post("/auth/register")
async def register(req: AuthRequest, request: Request):
    enforce_rate_limit(f"auth:{get_client_ip(request)}", config.RATE_LIMIT_AUTH)
    ok, msg = security.register_user(req.username, req.password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"detail": msg}


@app.post("/auth/login")
async def login(req: AuthRequest, request: Request):
    enforce_rate_limit(f"auth:{get_client_ip(request)}", config.RATE_LIMIT_AUTH)
    ok, result = security.login_user(req.username, req.password)
    if not ok:
        raise HTTPException(status_code=401, detail=result)
    return {"token": result, "token_type": "bearer"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    req:      ChatRequest,
    request:  Request,
    username: str = Depends(require_auth),
):
    enforce_rate_limit(f"chat:{username}:{get_client_ip(request)}", config.RATE_LIMIT_CHAT)

    clean = security.sanitize_input(req.message)
    ok, err = security.validate_message(clean)
    if not ok:
        raise HTTPException(status_code=400, detail=err)

    t0            = time.perf_counter()
    intent, score = bot.kb.find_best_intent(clean, bot.threshold)
    elapsed_ms    = (time.perf_counter() - t0) * 1000

    bot._stats["total"]    += 1
    bot._stats["time_ms"].append(elapsed_ms)

    if intent:
        bot._stats["matched"] += 1
        reply = bot.engine.get_response(intent)
        persistence.log_conversation(clean, reply, matched=True, score=round(score, 4))
        return ChatResponse(response=reply, matched=True, score=round(score, 4), mode="nlp")

    reply = bot.engine.fallback()
    persistence.log_conversation(clean, reply, matched=False, score=round(score, 4))
    return ChatResponse(response=reply, matched=False, score=round(score, 4), mode="nlp")


@app.post("/chat/upload", response_model=ChatResponse)
async def chat_upload(
    request:  Request,
    message:  str        = Form(default="Analizeaza fisierul."),
    file:     UploadFile = File(...),
    username: str        = Depends(require_auth),
):
    """
    Primeste un fisier text, extrage continutul si il trimite ca mesaj catre NLP.
    Suporta: .txt, .md, .py, .js, .json, .csv, .html, .css, .cpp, .c, .java,
             .xml, .yaml, .yml
    Imaginile si PDF-urile nu sunt procesate fara Claude API.
    """
    enforce_rate_limit(f"chat:{username}:{get_client_ip(request)}", config.RATE_LIMIT_CHAT)

    content = await file.read()
    fname   = file.filename or "fisier"
    mime    = file.content_type or ""

    ok, err = security.validate_file(fname, len(content), mime)
    if not ok:
        raise HTTPException(status_code=400, detail=err)

    
    ext = Path(fname).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".pdf"}:
        raise HTTPException(
            status_code=415,
            detail="Imaginile si PDF-urile necesita Claude API. Trimite fisiere text."
        )

    
    try:
        file_text = content.decode("utf-8", errors="replace")
    except Exception:
        raise HTTPException(status_code=400, detail="Fisierul nu poate fi citit ca text.")

    
    if len(file_text) > 2000:
        file_text = file_text[:2000] + "\n...[trunchiat]"

    
    clean_message = security.sanitize_input(message)
    combined      = f"{clean_message}\n\n[Continut {fname}]:\n{file_text}"

    ok, err = security.validate_message(combined[:config.MAX_MESSAGE_LENGTH])
    if not ok:
        raise HTTPException(status_code=400, detail=err)

    t0            = time.perf_counter()
    intent, score = bot.kb.find_best_intent(combined, bot.threshold)
    elapsed_ms    = (time.perf_counter() - t0) * 1000

    bot._stats["total"]    += 1
    bot._stats["time_ms"].append(elapsed_ms)

    if intent:
        bot._stats["matched"] += 1
        reply = bot.engine.get_response(intent)
        persistence.log_conversation(
            f"[upload:{fname}] {clean_message}", reply, matched=True, score=round(score, 4)
        )
        return ChatResponse(response=reply, matched=True, score=round(score, 4), mode="nlp")

    reply = bot.engine.fallback()
    persistence.log_conversation(
        f"[upload:{fname}] {clean_message}", reply, matched=False, score=round(score, 4)
    )
    return ChatResponse(response=reply, matched=False, score=round(score, 4), mode="nlp")


@app.get("/stats", response_model=StatsResponse)
def stats(username: str = Depends(require_auth)):
    return StatsResponse(**bot.stats)
