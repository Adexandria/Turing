from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
import pytest

from turing.api.app import app
from turing.api.schemas import PredictionRequest, PredictionResponse


@pytest.fixture
def client():
    """Fixture that provides a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_inference_engine():
    """Fixture that provides a mocked inference engine."""
    with patch('turing.api.app.inference_engine') as mock:
        yield mock


class TestHealthCheck:
    """Test suite for the health check endpoint."""
    
    def test_health_check_returns_ok(self, client):
        """Test that the health check endpoint returns status ok."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "message": "Turing Code Classification API is ready."
        }


class TestPredictEndpoint:
    """Test suite for the predict endpoint."""
    
    def test_predict_success_java(self, client, mock_inference_engine):
        """Test successful prediction for Java code."""
        # Setup mock
        mock_inference_engine.predict_payload.return_value = (
            np.array([0, 1]),  # raw predictions as numpy array
            ["class", "method"],  # labels
            "run_id_123",  # run_id
            "models:/CodeBERTa_java/Production"  # artifact
        )
        
        # Make request
        request_data = {
            "texts": ["public class Main", "public void test()"],
            "language": "java"
        }
        response = client.post("/predict", json=request_data)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "labels" in data
        assert "model_info" in data
        assert data["labels"] == ["class", "method"]
        assert data["model_info"]["language"] == "java"
    
    def test_predict_success_python(self, client, mock_inference_engine):
        """Test successful prediction for Python code."""
        # Setup mock
        mock_inference_engine.predict_payload.return_value = (
            np.array([1, 0]),  # raw predictions as numpy array
            ["function", "class"],  # labels
            "run_id_456",  # run_id
            "models:/CodeBERTa_python/Production"  # artifact
        )
        
        # Make request
        request_data = {
            "texts": ["def main():", "class MyClass:"],
            "language": "python"
        }
        response = client.post("/predict", json=request_data)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["labels"] == ["function", "class"]
        assert data["model_info"]["language"] == "python"
    
    def test_predict_success_pharo(self, client, mock_inference_engine):
        """Test successful prediction for Pharo code."""
        # Setup mock
        mock_inference_engine.predict_payload.return_value = (
            np.array([0]),  # raw predictions as numpy array
            ["method"],  # labels
            "run_id_789",  # run_id
            "models:/CodeBERTa_pharo/Production"  # artifact
        )
        
        # Make request
        request_data = {
            "texts": ["initialize"],
            "language": "pharo"
        }
        response = client.post("/predict", json=request_data)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["labels"] == ["method"]
        assert data["model_info"]["language"] == "pharo"
    
    def test_predict_missing_texts(self, client):
        """Test that prediction fails when texts are missing."""
        request_data = {
            "language": "java"
        }
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_missing_language(self, client):
        """Test that prediction fails when language is missing."""
        request_data = {
            "texts": ["public class Main"]
        }
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_empty_texts(self, client, mock_inference_engine):
        """Test prediction with empty texts list."""
        mock_inference_engine.predict_payload.return_value = (
            np.array([]),  # raw predictions as empty numpy array
            [],  # labels
            "run_id_000",  # run_id
            "models:/CodeBERTa_java/Production"  # artifact
        )
        
        request_data = {
            "texts": [],
            "language": "java"
        }
        response = client.post("/predict", json=request_data)
        
        # Should succeed with empty results
        assert response.status_code == 200
        data = response.json()
        assert data["predictions"] == []
        assert data["labels"] == []
    
    def test_predict_error_handling(self, client, mock_inference_engine):
        """Test that prediction endpoint handles errors gracefully."""
        # Setup mock to raise an exception
        mock_inference_engine.predict_payload.side_effect = Exception("Model loading failed")
        
        request_data = {
            "texts": ["public class Main"],
            "language": "java"
        }
        response = client.post("/predict", json=request_data)
        
        # Should return 500 error
        assert response.status_code == 500
        assert "Model loading failed" in response.json()["detail"]
    
    def test_predict_invalid_language(self, client, mock_inference_engine):
        """Test prediction with invalid language parameter."""
        # The model might raise an error for unsupported language
        mock_inference_engine.predict_payload.side_effect = ValueError("Unsupported language: cobol")
        
        request_data = {
            "texts": ["IDENTIFICATION DIVISION."],
            "language": "cobol"
        }
        response = client.post("/predict", json=request_data)
        
        # Should return 500 error
        assert response.status_code == 500
        assert "Unsupported language" in response.json()["detail"]


class TestAPISchemas:
    """Test suite for API schemas validation."""
    
    def test_prediction_request_valid(self):
        """Test that PredictionRequest validates correct data."""
        request = PredictionRequest(
            texts=["public void main"],
            language="java"
        )
        assert request.texts == ["public void main"]
        assert request.language == "java"
    
    def test_prediction_response_valid(self):
        """Test that PredictionResponse validates correct data."""
        response = PredictionResponse(
            predictions=[0, 1],
            labels=["class", "method"],
            model_info={"artifact": "models:/CodeBERTa_java/Production", "language": "java"}
        )
        assert response.predictions == [0, 1]
        assert response.labels == ["class", "method"]
        assert response.model_info["language"] == "java"
