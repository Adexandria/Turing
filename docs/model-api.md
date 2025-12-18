# Model API

The project provides a **REST API** built with **FastAPI**, exposing a **`/predict` endpoint** for classifying code comment on its programming language.  
This API allows developers to send text snippets and receive predicted labels along with model metadata.

----------

## Test the API Locally

!!! tip  
    Make sure the **git repository has been cloned** and all dependencies are installed before running the server.

### 1. Start the Server

Run the FastAPI server locally using **Uvicorn**:

```bash
uvicorn turing.api.app:app --reload
```

-   The `--reload` flag enables automatic reloading whenever code changes are detected.
    
-   By default, the server runs at `http://127.0.0.1:8000`.
    

----------

### 2. Request Payload

Send a JSON request containing:

-   `texts`: a list of code comments
    
-   `language`: the language context for prediction
    

``` json
{  
    "texts":  [ 
         "public void main",
        "def init self" 
         ],  
    "language":  "pharo" 
}
```

----------

### Response Body

The API returns a JSON response with:

-   `labels`: predicted labels for each input text
-  `predictions`: numeric labels for each input text
-   `model_info`: metadata about the model used for prediction
    
###   Response body

``` json
{
  "predictions": [
    [
      1,0,0,0,0,0,0
    ],
    [
      1,0,0,0,0,0,0
    ],
  ],
  "labels": [
    [
      "summary"
    ],
    [
      "summary"
    ]
  ],"model_info": {
    "artifact": "RandomForestTfIdf_java",
    "language": "java"
  }
}
```