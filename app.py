import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page title
st.set_page_config(page_title="CKD Prediction App", layout="wide")

# Load the trained model and label encoders
@st.cache_resource
def load_models():
    try:
        with open('models/ckd_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('models/label_encoders.pkl', 'rb') as file:
            label_encoders = pickle.load(file)
        return model, label_encoders
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def process_input(input_data, label_encoders):
    """Process input data by encoding categorical variables"""
    processed_data = input_data.copy()
    
    # Encode categorical variables
    categorical_columns = ['rbc', 'pc', 'pcc', 'ba', 'htn']
    for column in categorical_columns:
        if column in processed_data.columns:
            processed_data[column] = label_encoders[column].transform(processed_data[column])
    
    return processed_data

def plot_feature_importances(model, feature_names):
    """Plot feature importances"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    st.pyplot(plt)

def main():
    # Add custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 20px;
        }
        .stButton>button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # App title and description
    st.title('🏥 Chronic Kidney Disease Prediction')
    st.write('Enter patient information to predict CKD risk')
    
    try:
        # Load models
        model, label_encoders = load_models()
        
        if model is not None and label_encoders is not None:
            # Create two columns for input fields
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Patient Information")
                age = st.number_input('Age', min_value=0, max_value=100, value=40)
                blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200, value=80)
                specific_gravity = st.number_input('Specific Gravity', min_value=1.005, max_value=1.025, value=1.015, step=0.001, format="%.3f")
                albumin = st.selectbox('Albumin', options=[0, 1, 2, 3, 4, 5])
                sugar = st.selectbox('Sugar', options=[0, 1, 2, 3, 4, 5])
            
            with col2:
                st.subheader("Additional Parameters")
                red_blood_cells = st.selectbox('Red Blood Cells', ['normal', 'abnormal'])
                pus_cell = st.selectbox('Pus Cell', ['normal', 'abnormal'])
                pus_cell_clumps = st.selectbox('Pus Cell Clumps', ['present', 'notpresent'])
                bacteria = st.selectbox('Bacteria', ['present', 'notpresent'])
                hypertension = st.selectbox('Hypertension', ['yes', 'no'])
            
            # Add prediction button
            if st.button('Predict', key='predict'):
                # Create input data DataFrame
                input_data = pd.DataFrame({
                    'age': [age],
                    'bp': [blood_pressure],
                    'sg': [specific_gravity],
                    'al': [albumin],
                    'su': [sugar],
                    'rbc': [red_blood_cells],
                    'pc': [pus_cell],
                    'pcc': [pus_cell_clumps],
                    'ba': [bacteria],
                    'htn': [hypertension]
                })
                
                # Process input data
                processed_data = process_input(input_data, label_encoders)
                
                # Make prediction
                prediction = model.predict(processed_data)
                prediction_proba = model.predict_proba(processed_data)
                
                # Display result
                st.markdown("---")
                st.subheader("Prediction Result")
                
                if prediction[0] == 1:
                    st.error('🚨 High Risk of Chronic Kidney Disease')
                    st.write(f'Confidence: {prediction_proba[0][1]:.2%}')
                else:
                    st.success('✅ Low Risk of Chronic Kidney Disease')
                    st.write(f'Confidence: {prediction_proba[0][0]:.2%}')
                
                # Add disclaimer
                st.markdown("---")
                st.caption("⚠️ Disclaimer: This is a prediction tool and should not be used as a substitute for professional medical advice.")
            
            # Plot feature importances
            st.markdown("---")
            st.subheader("Feature Importances")
            feature_names = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn']
            plot_feature_importances(model, feature_names)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure all model files are properly uploaded to the repository.")

if __name__ == '__main__':
    main()
