from sentence_transformers import SentenceTransformer
import mlflow.pyfunc
import joblib

class MiniLMClassifierWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.encoder = SentenceTransformer(context.artifacts["encoder_path"])
        self.classifier = joblib.load(context.artifacts["classifier_path"])

    def predict(self, context, model_input):
        embeddings = self.encoder.encode(model_input)
        predictions = self.classifier.predict(embeddings)
        return predictions