# Git Agent (Local)

This project runs locally with:

- FastAPI backend in `Backend/api_server.py`
- Next.js frontend in `Frontend/`
- Repository analysis agent in `Backend/LLMHandler/`

## Local Run

Backend:

```bash
cd /Users/ramanpandey/Developer/git-agent
source venv/bin/activate
python -m uvicorn Backend.api_server:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd /Users/ramanpandey/Developer/git-agent/Frontend
npm install
npm run dev -- --hostname 127.0.0.1 --port 3000
```

Open:

- Frontend: http://127.0.0.1:3000
- Backend health: http://127.0.0.1:8000/health

## Notes

- Use `127.0.0.1` consistently (avoid mixing with `localhost`).
- If UI styles look wrong, hard refresh once.
- The agent now uses dynamic query-aware prompt guidance and better evidence retrieval for specific questions (for example, database-identification questions).
