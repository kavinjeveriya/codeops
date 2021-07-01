from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
model = Sequential()
model.add(Convolution2D(filters=64, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=3, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
from keras_preprocessing.image import ImageDataGenerator
traingen = ImageDataGenerator(rescale=1./255 , zoom_range=0.2 , horizontal_flip=True ,  vertical_flip=True )
testgen = ImageDataGenerator(rescale=1./255)
trainset = traingen.flow_from_directory('/code/cnn_dataset/training_set/' , target_size=(64,64) , batch_size=32 , class_mode='categorical' )
testset = testgen.flow_from_directory('/code/cnn_dataset/test_set/' , target_size=(64,64) , batch_size=32 , class_mode='categorical' )
model.fit(trainset , epochs=3 , validation_data=testset , steps_per_epoch=9259, validation_steps=10  )
model.save("cnnmlops.h5")
scores = model.evaluate(testset,verbose=1)
print('loss',scores[0])
print('acc',scores[1])
acc=scores[1]*100
file = open("/code/accuracy.txt", "w")
file.write(str(acc))
file.close()






