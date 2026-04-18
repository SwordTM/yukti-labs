from __future__ import annotations

import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ingestion import ingest_paper
from ingestion.arxiv_resolver import ArxivResolverError
from ingestion.component_extractor import ComponentExtractorError
from schema.models import ComponentManifest

load_dotenv()

logger = logging.getLogger("ml_lens.backend")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="ML Lens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    url: str = Field(..., description="arXiv URL or bare id (e.g. 1706.03762)")


class IngestResponse(BaseModel):
    manifest: ComponentManifest


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    try:
        manifest = ingest_paper(req.url)
    except ArxivResolverError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ComponentExtractorError as exc:
        logger.exception("component extraction failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("unexpected ingestion error")
        raise HTTPException(status_code=500, detail=f"ingestion failed: {exc}") from exc
    return IngestResponse(manifest=manifest)
