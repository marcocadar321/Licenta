# ── Etapa 1: compilare C++ ───────────────────────────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY preprocess.cpp .
RUN g++ -O2 -shared -fPIC -o preprocess.so preprocess.cpp


# ── Etapa 2: aplicație ───────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Biblioteca C++ compilată
COPY --from=builder /build/preprocess.so ./build/preprocess.so

# Dependințe Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codul sursă și datele
COPY config.py security.py persistence.py nlp.py claude_client.py api.py main.py ./
COPY data/ ./data/
COPY static/ ./static/

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
