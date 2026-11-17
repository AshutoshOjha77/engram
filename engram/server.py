from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import engram

app = FastAPI(title="Engram REST API")


class RememberRequest(BaseModel):
    text: str
    dataset: str = "default"
    strength: float = Field(default=1.0, ge=0.0, le=1.0)


class RecallRequest(BaseModel):
    query: str
    dataset: str | None = None
    datasets: list[str] | None = None
    top_k: int | None = Field(default=None, gt=0)


class ForgetRequest(BaseModel):
    dataset: str


class PruneRequest(BaseModel):
    dataset: str = "default"
    threshold: float = 0.2


class ImproveRequest(BaseModel):
    text: str
    dataset: str = "default"


@app.post("/remember")
async def api_remember(req: RememberRequest) -> dict:
    try:
        await engram.remember(text=req.text, dataset=req.dataset, strength=req.strength)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"status": "ok", "dataset": req.dataset}


@app.post("/recall")
async def api_recall(req: RecallRequest) -> dict:
    target_datasets = req.datasets
    if target_datasets is None and req.dataset is not None:
        target_datasets = [req.dataset]

    results = await engram.recall(query=req.query, datasets=target_datasets)
    if req.top_k is not None:
        results = results[: req.top_k]
    return {"results": results}


@app.post("/forget")
async def api_forget(req: ForgetRequest) -> dict:
    await engram.forget(dataset=req.dataset)
    return {"status": "ok", "dataset": req.dataset}


@app.post("/prune")
async def api_prune(req: PruneRequest) -> dict:
    pruned_count = await engram.prune(dataset=req.dataset, threshold=req.threshold)
    return {"status": "ok", "dataset": req.dataset, "pruned": pruned_count}


@app.post("/improve")
async def api_improve(req: ImproveRequest) -> dict:
    await engram.improve(text=req.text, dataset=req.dataset)
    return {"status": "ok", "dataset": req.dataset}
