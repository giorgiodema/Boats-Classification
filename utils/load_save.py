import keras
import os

def load_model(filename):
    if not os.path.exists(filename):
        print("Model not found")
        return None
    else:
        model = keras.models.load_model(filename)
        print("Model loaded successfully")
        return model
    
def save_model(model, filename):
    keras.models.save_model(model,filename)
    print("Model saved")