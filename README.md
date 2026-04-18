# ML Lens

A collaborative ML evaluation and testing platform built with FastAPI, React, and DeepEval.

## Project Structure

```
ml-lens/
├── backend/          # FastAPI + Python backend
├── frontend/         # React + Vite frontend
├── evals/           # DeepEval + pytest evaluation suite
└── shared/          # Shared schema types (JSON)
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- Node.js 18+
- API Keys: ANTHROPIC_API_KEY, E2B_API_KEY, LANGFUSE_KEY

### Setup

1. Clone the repo
```bash
git clone https://github.com/rov33r/yukti-labs_florentxlundaisociety.git
cd yukti-labs_florentxlundaisociety
```

2. Copy environment variables
```bash
cp .env.example .env
# Add your API keys to .env
```

3. Start services
```bash
docker-compose up
```

- Backend: http://localhost:8000
- E2B Sandbox: http://localhost:4242

## Development Workflow

### Creating a Feature
1. Create a branch: `git checkout -b feature/your-feature`
2. Make changes
3. Open a PR → review → merge to main

### Backend Development
```bash
cd ml-lens/backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend Development
```bash
cd ml-lens/frontend
npm install
npm run dev
```

### Running Evaluations
```bash
cd ml-lens/evals
pytest
```

## Tech Stack
- **Backend**: FastAPI, Python
- **Frontend**: React, Vite
- **Testing**: DeepEval, pytest
- **Sandbox**: E2B
- **LLM**: Anthropic Claude API
- **Observability**: Langfuse

