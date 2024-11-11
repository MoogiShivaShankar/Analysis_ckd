import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def train_and_save_model():
    # Sample data structure (replace this with your actual data loading)
    data = pd.DataFrame({
        'age': np.random.randint(20, 80, 1000),
        'bp': np.random.randint(60, 180, 1000),
        'sg': np.random.uniform(1.005, 1.025, 1000),
        'al': np.random.randint(0, 6, 1000),
        'su': np.random.randint(0, 6, 1000),
        'rbc': np.random.choice(['normal', 'abnormal'], 1000),
        'pc': np.random.choice(['normal', 'abnormal'], 1000),
        'pcc': np.random.choice(['present', 'notpresent'], 1000),
        'ba': np.random.choice(['present', 'notpresent'], 1000),
        'htn': np.random.choice(['yes', 'no'], 1000),
        'classification': np.random.randint(0, 2, 1000)  # Target variable
    })
    
    # Separate features and target
    X = data.drop('classification', axis=1)
    y = data['classification']
    
    # Initialize dictionary to store label encoders
    label_encoders = {}
    
    # Encode categorical variables
    categorical_columns = ['rbc', 'pc', 'pcc', 'ba', 'htn']
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save model and encoders
    with open('models/ckd_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    with open('models/label_encoders.pkl', 'wb') as file:
        pickle.dump(label_encoders, file)
    
    print("Model and encoders saved successfully!")

if __name__ == "__main__":
    train_and_save_model()