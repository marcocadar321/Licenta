# ChatBot NLP v6

Chatbot semantic Python + C++ cu autentificare JWT, rate limiting, sanitizare input și Claude AI.

## Structura proiectului

```
chatbot/
├── config.py          # Toate setările (citite din .env)
├── security.py        # JWT, hash parole, rate limiting, sanitizare
├── persistence.py     # Stocare conversații MongoDB (opțional)
├── nlp.py             # Motor NLP local (sentence-transformers + C++)
├── claude_client.py   # Client async Claude API
├── api.py             # Server FastAPI (web + upload)
├── main.py            # Interfață terminal
├── preprocess.cpp     # Normalizare text C++ (opțional)
│
├── data/
│   ├── knowledge_base.json   # Intenții și răspunsuri
│   └── users.json            # Conturi (creat automat)
│
├── static/
│   └── index.html            # UI chatbot
│
├── build/                    # preprocess.so / .dll (generat)
├── .env.example              # Șablon variabile de mediu
├── requirements.txt
├── Dockerfile
└── render.yaml
```

## Instalare rapidă

```bash
# 1. Clonează / extrage arhiva
cd chatbot/

# 2. Instalează dependințele
pip install -r requirements.txt

# 3. Configurează variabilele de mediu
cp .env.example .env
# Editează .env și adaugă cel puțin ANTHROPIC_API_KEY și SECRET_KEY

# 4. (Opțional) Compilează modulul C++ pentru performanță mai bună
mkdir -p build
# Linux / macOS:
g++ -O2 -shared -fPIC -o build/preprocess.so preprocess.cpp
# Windows:
g++ -O2 -shared -o build/preprocess.dll preprocess.cpp
```

## Activare chatbot

### Terminal (mod conversație directă)

```bash
# Mod NLP local (fără API key)
python main.py

# Mod Claude AI (necesită ANTHROPIC_API_KEY în .env)
python main.py --claude
```

Tastează mesaje direct. Scrie `exit` sau `quit` pentru a ieși.

### Server web

```bash
# Mod development (cu auto-reload)
uvicorn api:app --reload --port 8000

# Mod producție
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 2
```

Deschide `http://localhost:8000` în browser, creează un cont și începe conversația.

### Docker

```bash
# Build
docker build -t chatbot-nlp .

# Rulare cu variabile de mediu
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e SECRET_KEY=cheie-secreta-lunga \
  chatbot-nlp
```

### Deploy pe Render.com

1. Push codul pe GitHub
2. Conectează repo-ul în [render.com](https://render.com)
3. Render detectează automat `render.yaml`
4. Adaugă `ANTHROPIC_API_KEY` și `SECRET_KEY` în **Dashboard > Environment**
5. Deploy pornit automat

## Configurare (.env)

| Variabilă | Obligatoriu | Descriere |
|-----------|-------------|-----------|
| `SECRET_KEY` | ✅ în producție | Cheie semnare JWT (min 32 caractere) |
| `ANTHROPIC_API_KEY` | Recomandat | Activează modul Claude AI |
| `MONGODB_URI` | ❌ opțional | Stocare conversații MongoDB Atlas |
| `TOKEN_EXPIRE_SECONDS` | ❌ | Durata token JWT (default 3600) |
| `RATE_LIMIT_CHAT` | ❌ | Mesaje max/minut/user (default 30) |
| `RATE_LIMIT_AUTH` | ❌ | Încercări login/minut/IP (default 10) |
| `ALLOWED_ORIGINS` | ❌ | CORS – în producție pune domeniul tău |
| `CLAUDE_MODEL` | ❌ | Modelul Claude (default: claude-opus-4-6) |

## Cum funcționează

```
Mesaj utilizator
       │
       ▼
  sanitizare + validare
       │
       ▼
  ANTHROPIC_API_KEY setat?
   ├─ DA  → Claude API → răspuns
   └─ NU  → NLP local:
              BERT fine-tunat (dacă există models/)
              └─ fallback → sentence-transformers
                             └─ fallback → răspuns implicit
```

## Securitate implementată

- **JWT** HMAC-SHA256, expiră după `TOKEN_EXPIRE_SECONDS`
- **Parole** PBKDF2-SHA256 cu salt (260.000 iterații)
- **Rate limiting** fereastră glisantă per user+IP
- **Sanitizare XSS** strip `<script>`, `javascript:`, `on*=`
- **Validare fișiere** extensie + MIME + dimensiune

## Extindere knowledge base

Editează `data/knowledge_base.json`:

```json
{
  "intents": [
    {
      "tag": "salut",
      "patterns": ["bună", "salut", "hey", "bună ziua"],
      "responses": ["Bună! Cu ce te pot ajuta?", "Salut! Ce dorești?"]
    }
  ]
}
```

Repornește serverul după modificări.
