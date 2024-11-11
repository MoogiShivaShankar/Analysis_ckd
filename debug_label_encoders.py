def check_label_encoders():
    """Debug function to check label encoders"""
    try:
        with open('models/label_encoders.pkl', 'rb') as file:
            encoders = pickle.load(file)
            
        print("Type of label_encoders:", type(encoders))
        
        if isinstance(encoders, dict):
            print("\nAvailable encoders:")
            for key, encoder in encoders.items():
                print(f"\n{key}:")
                print("Type:", type(encoder))
                print("Classes:", encoder.classes_)
        else:
            print("Warning: encoders is not a dictionary!")
            print("Content:", encoders)
            
    except Exception as e:
        print(f"Error loading encoders: {str(e)}")

# Run this function to debug
check_label_encoders()