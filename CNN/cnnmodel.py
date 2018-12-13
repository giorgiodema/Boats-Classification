from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Model

img_shape = (240,800,3)

class ConvolutionalNN:
    def __init__(self,img_shape, num_cat):
        
        # The input layer is a matrix with shape(240,800,3)
        x = Input(shape=img_shape)
        # 64 filters --> shape = (240,800)
        f = Conv2D(filters=64, kernel_size=(24,80),activation='relu')(x)
        # MaxPooling --> shape = (120,400)
        p = MaxPool2D(pool_size=(2,2))(f)
        # 16 filters --> shape = (120,400)
        f2 = Conv2D(filters=16,kernel_size=(6,20),activation='relu')(p)
        # MaxPooling --> shape = (30,100)
        p2 = MaxPool2D(pool_size=(4,4))(f2)
        # Flatten    --> shape = (3000)
        fl = Flatten()(p2)
        d1 = Dense(units=1000,activation='relu')(fl)
        d2 = Dense(units=500,activation='relu')(d1)
        output = Dense(units=num_cat,activation='softmax')(d2)