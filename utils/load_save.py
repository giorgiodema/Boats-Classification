import keras
import os

MODELNAME = "inceptionV3.h5"

def load_model():
    filename = os.path.join(MODELNAME)
    if not os.path.exists(filename):
        print("Model not found")
        return None
    else:
        model = keras.models.load_model(MODELNAME)
        print("Model loaded successfully")
        return model
    
def save_model(model):
    filename = os.path.join(MODELNAME)
    keras.models.save_model(model,filename)
    print("Model saved")