from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

app = FastAPI()

# Load the model
model = tf.keras.models.load_model("app/model/earthquake.h5")

# Define a Pydantic model for input validation
class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    depth: float
    magnitude: float
    year: int
    damage_description: str  # Assuming this is one of the features used in the model (or you can remove if not)

# Define the predict endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    # Prepare the input data
    input_data = np.array([
        [
            request.latitude,
            request.longitude,
            request.depth,
            request.magnitude,
            request.year,
            1 if request.damage_description else 0  # Binary encoding for damage_description
        ]
    ])
    
    # Ensure that the input has the shape (1, 6)
    print(f"Input data shape: {input_data.shape}")  # For debugging

    # Get the model's prediction
    prediction = model.predict(input_data)
    
    # Return the prediction
    return {"prediction": prediction.tolist()}
