import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
import pathlib
from PIL import Image
import seaborn as sns 
import yaml
import keras 
import pandas as pd 

# Importation du fichier yaml 
with open('app.yaml') as yaml_data:
    params = yaml.safe_load(yaml_data)
print(params)

# Création des constantes 
IMAGE_HEIGHT = params['image']['height']
IMAGE_WIDTH = params['image']['width']
IMAGE_DEPTH = params['image']['depth']
MODEL_DIR = pathlib.Path(params['dir']['model'])  # à revoir 
DATA_DIR = pathlib.Path(params['dir']['data'])

def load_image(path):
    """Load an image as numpy array
    """
    return plt.imread(path)
    

def predict_image(path, model):
    """Predict plane identification from image.
    
    Parameters
    ----------
    path (Path): path to image to identify
    model (keras.models): Keras model to be used for prediction
    
    Returns
    -------
    Predicted class
    """
    images = np.array([np.array(Image.open(path).resize((IMAGE_WIDTH, IMAGE_HEIGHT)))])
    print(images.shape)
    prediction_vector = model.predict(images)
    predicted_classes = np.argmax(prediction_vector, axis=1)
    return predicted_classes[0]

def proba(path, model):
    """Compute the probability of prediction plane identification from image.
    
    Parameters
    ----------
    path (Path): path to image to identify
    model (keras.models): Keras model to be used for prediction
    
    Returns
    -------
    Probability of model prediction
    """
    images = np.array([np.array(Image.open(path).resize((IMAGE_WIDTH, IMAGE_HEIGHT)))])
    print(images.shape)
    prediction_vector = model.predict(images)
    prediction_proba = max(prediction_vector)
    return prediction_proba[0]

def barplotting(path, model):
    """Show probability of prediction plane identification in a bar chart.
    
    Parameters
    ----------
    path (Path): path to image to identify
    model (keras.models): Keras model to be used for prediction
    
    Returns
    -------
    Probability of model prediction
    """
    images = np.array([np.array(Image.open(path).resize((IMAGE_WIDTH, IMAGE_HEIGHT)))])
    # print(images.shape)
    prediction_vector = model.predict(images)
    predicted_classes = np.argmax(prediction_vector, axis=1)[0]
    st.bar_chart(prediction_vector)
    # return barplot

def load_model(path):
    """Load tf/Keras model for prediction
    """
    return tf.keras.models.load_model(path)

model = load_model(MODEL_DIR)
model.summary()

st.title("Identification d'avion")

uploaded_file = st.file_uploader("Charger une image d'avion") #, accept_multiple_files=True)#

if uploaded_file:
    loaded_image = load_image(uploaded_file)
    st.image(loaded_image)
    
predict_btn = st.button("Identifier", disabled=(uploaded_file is None))
if predict_btn:
    prediction = predict_image(uploaded_file, model)
    st.write(f"C'est un : {prediction}")
    # Exemple si les f-strings ne sont pas dispo.
    #st.write("C'est un : {}".format(prediction)
    # Afficher la probabilité de la prédiction du modèle
    probability = proba(uploaded_file, model)
    st.write(f"La probabilité associée à la prédiction du modèle est de : {probability}")

barchat_btn = st.button("Afficher les probabilités", disabled=(uploaded_file is None))
if barchat_btn:
    st.write(f"La probabilité associée à la prédiction du modèle est de")
    graph=barplotting(uploaded_file, model)