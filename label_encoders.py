from sklearn.preprocessing import LabelEncoder
import pickle

# Create and save label encoders properly
def create_and_save_encoders(df):
    # Dictionary to store label encoders
    label_encoders = {}
    
    # Categorical columns that need encoding
    categorical_columns = ['rbc', 'pc', 'pcc', 'ba', 'htn']
    
    # Create and fit label encoders
    for column in categorical_columns:
        if column in df.columns:
            label_encoders[column] = LabelEncoder()
            label_encoders[column].fit(df[column])
    
    # Save label encoders as dictionary
    with open('models/label_encoders.pkl', 'wb') as file:
        pickle.dump(label_encoders, file)
    
    return label_encoders