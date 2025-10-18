from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class RecognizeItem(BaseModel):
    is_returning: bool
    similarity: float
   

class RecognizeResponse(BaseModel):
    num_faces: int
    results: list[RecognizeItem]


class RecognizePerImage(BaseModel):
    filename: str
    num_faces: int
    results: list[RecognizeItem]


class RecognizeBatchResponse(BaseModel):
    items: list[RecognizePerImage]


