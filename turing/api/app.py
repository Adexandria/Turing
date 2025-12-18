import base64
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import gradio as gr
from loguru import logger

from turing.api.demo import create_demo
from turing.api.schemas import PredictionRequest, PredictionResponse
from turing.modeling.predict import ModelInference


def get_logo_b64_src(filename="logo_header.svg"):
    """read SVG and convert it into a string Base64 for HTML."""
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.join(base_path, "..", "..", "reports", "figures", filename)
        target_path = os.path.normpath(target_path)
        
        with open(target_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/svg+xml;base64,{encoded}"
    except Exception as e:
        print(f"Unable to load logo for API: {e}")
        return "" 


# load logo
logo_src = get_logo_b64_src()

# html
logo_html_big = f"""
<a href="/gradio">
    <img src="{logo_src}" width="150" style="display: block; margin: 10px 0;">
</a>
"""

# description
description_md = f"""
API for classifying code comments.

You can interact with the model directly using the visual interface. 
Click the logo below to open it:

{logo_html_big}

"""

app = FastAPI(
    title="Turing Team Code Classification API",
    description=description_md,
    version="1.0.0"
)

@app.get("/manifest.json")
def get_manifest():
    return JSONResponse(content={
        "name": "Turing App",
        "short_name": "Turing",
        "start_url": "/gradio",
        "display": "standalone",
        "background_color": "#ffffff",
        "theme_color": "#000000",
        "icons": []
    })

# Global inference engine instance
inference_engine = ModelInference()

demo = create_demo(inference_engine)
app = gr.mount_gradio_app(app, demo, path="/gradio")

@app.get("/")
def health_check():
    """
    Root endpoint to verify API status.
    """
    return {"status": "ok", "message": "Turing Code Classification API is ready.", "ui_url": "/gradio"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Endpoint to classify a list of code comments.
    Dynamically loads the model from MLflow based on the request parameters.
    """
    try:
        logger.info(f"Received prediction request for language: {request.language}")

        # Perform prediction using the inference engine
        raw, predictions, run_id, artifact = inference_engine.predict_payload(
            texts=request.texts, language=request.language
        )

        # Ensure predictions are serializable (convert numpy arrays to lists)
        if hasattr(predictions, "tolist"):
            predictions = predictions.tolist()

        return PredictionResponse(
            predictions=raw.tolist(),
            labels=predictions,
            model_info={"artifact": artifact, "language": request.language},
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Entry point for running the API directly with python
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7860)
