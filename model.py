import csv
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np

csvFile = './data/driving_log.csv'

lines = []
with open( csvFile ) as input:
    reader = csv.reader( input )
    for line in reader:
        lines.append( line )

lines = lines[1:]

train_samples, validation_samples = train_test_split( lines, test_size=0.2 )

def generator( samples, batch_size=32 ):
    num_samples = len( samples )

    while 1:
        sklearn.utils.shuffle( samples )

        for offset in range( 0, num_samples, batch_size ):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                centerFile = batch_sample[0].split('/')[-1]
                leftFile = batch_sample[1].split('/')[-1]
                rightFile = batch_sample[2].split('/')[-1]
               
                centerFileName = './data/IMG/' + centerFile
                leftFileName = './data/IMG/' + leftFile
                rightFileName = './data/IMG/' + rightFile
             
                centerImage = mpimg.imread( centerFileName )
                leftImage = mpimg.imread( leftFileName )
                rightImage = mpimg.imread( rightFileName )

                image_flipped = np.copy( np.fliplr( centerImage ) )
                
                images.append( centerImage )
                images.append( leftImage )
                images.append( rightImage )
                images.append( image_flipped )
                
                correction = 0.08
                angle_center = float( batch_sample[3] )
                angle_left = angle_center + correction
                angle_right = angle_center - correction

                angle_flipped = -angle_center
                
                angles.append( angle_center )
                angles.append( angle_left ) 
                angles.append( angle_right )
                angles.append( angle_flipped )

            X_train = np.array( images )
            y_train = np.array( angles )
           

            yield sklearn.utils.shuffle( X_train, y_train )

print( len( train_samples ) )
print( len( validation_samples ) )

train_generator = generator( train_samples, batch_size=32 )
validation_generator = generator( validation_samples, batch_size=32 )

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D

model = Sequential()

model.add( Cropping2D( cropping=( (50,20), (0,0) ), input_shape=(160,320,3)))

model.add( Lambda( lambda x: x/255. - 0.5 ) )

model.add( Convolution2D( 24, 5, 5, subsample=(2,2), activation = 'relu' ) )
model.add( Convolution2D( 36, 5, 5, subsample=(2,2), activation = 'relu' ) )
model.add( Convolution2D( 48, 5, 5, subsample=(2,2), activation = 'relu' ) )
model.add( Convolution2D( 64, 3, 3, subsample=(1,1), activation = 'relu' ) )
model.add( Convolution2D( 64, 3, 3, subsample=(1,1), activation = 'relu' ) )

model.add( Flatten() )

model.add( Dense( 100 ) )
model.add(Dropout(0.5)) 
model.add( Dense( 50 ) )
model.add( Dense( 10 ) )
model.add( Dense( 1 ) )

model.compile( loss='mse', optimizer='adam' )

train_steps = np.ceil( len( train_samples )/32 ).astype( np.int32 )
validation_steps = np.ceil( len( validation_samples )/32 ).astype( np.int32 )

model.fit_generator( train_generator, \
    steps_per_epoch = train_steps, \
    epochs=5, \
    verbose=1, \
    callbacks=None, 
    validation_data=validation_generator, \
    validation_steps=validation_steps, \
    class_weight=None, \
    max_q_size=10, \
    workers=1, \
    pickle_safe=False, \
    initial_epoch=0 )

model.save( 'model.h5' )
