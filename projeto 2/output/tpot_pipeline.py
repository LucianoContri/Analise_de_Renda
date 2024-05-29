import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Função para carregar o modelo treinado
def load_model(model_path):
    global exported_pipeline
    exported_pipeline = joblib.load(model_path)

# Função de previsão
def predict(data):
    return exported_pipeline.predict(data)