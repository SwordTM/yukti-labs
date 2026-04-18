from fastapi import FastAPI

app = FastAPI(title="ML Lens API")

@app.get("/health")
async def health():
    return {"status": "healthy"}
