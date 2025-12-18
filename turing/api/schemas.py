from typing import Any, List

from pydantic import BaseModel, Field


# Input Schema
class PredictionRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        description="List of code comments to classify",
        example=["public void main", "def init self"],
    )
    language: str = Field(
        ..., description="Programming language (java, python, pharo)", example="java"
    )


# Output Schema
class PredictionResponse(BaseModel):
    predictions: List[Any] = Field(..., description="List of predicted labels")
    labels: List[Any] = Field(..., description="List of human-readable labels")
    model_info: dict = Field(..., description="Metadata about the model used")
