from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io

# Load the pre-trained model
model = tf.keras.models.load_model('dog_vs_cat_classifier.keras')

# Initialize FastAPI app
app = FastAPI()

# Function to preprocess the uploaded image
def preprocess_image(image, img_size=(150, 150)):
    image = image.resize(img_size)  # Resize image to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    if image.shape[-1] == 4:  # Convert RGBA to RGB if necessary
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict the uploaded image
def predict_image(image, model):
    image = preprocess_image(image)  # Preprocess the uploaded image
    prediction = model.predict(image)  # Make prediction
    return prediction

# Simple GET method to check if the API is running
@app.get("/")
async def root():
    return {"message": "Welcome to the Dog vs Cat Classifier API!"}

# POST method: Upload image and predict
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image file and open it using PIL
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Predict the image
    prediction = predict_image(image, model)

    # Interpretation of the result
    if prediction[0] > 0.5:
        label = "Dog"
        confidence = prediction[0][0] * 100
    else:
        label = "Cat"
        confidence = (1 - prediction[0][0]) * 100

    # Return prediction result as JSON
    return JSONResponse(content={"label": label, "confidence": f"{confidence:.2f}%"})