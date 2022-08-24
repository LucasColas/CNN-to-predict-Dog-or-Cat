import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

print(tf.__version__)

#Data Preprocessing

#Preprocessing the training set
train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True
)

training_set = train_datagen.flow_from_directory(
                'C:\\Users\\lucas\\Desktop\\Deep Learning\\Deep Learning A-Z\\Machine Learning A-Z (Codes and Datasets)\\Part 8 - Deep Learning\\Section 40 - Convolutional Neural Networks (CNN)\\Python\\Section 40 - Convolutional Neural Networks (CNN)\\dataset\\training_set',
                target_size=(64,64),
                batch_size=32,
                class_mode='binary'
)

#Preprocessing the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
            'C:\\Users\\lucas\\Desktop\\Deep Learning\\Deep Learning A-Z\\Machine Learning A-Z (Codes and Datasets)\\Part 8 - Deep Learning\\Section 40 - Convolutional Neural Networks (CNN)\\Python\\Section 40 - Convolutional Neural Networks (CNN)\\dataset\\test_set',
            target_size=(64,64),
            batch_size=32,
            class_mode='binary'
)

#CNN
cnn = tf.keras.models.Sequential()
#Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=(64,64,3))) #filters : number of features to be detected

#Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

#Second ConvNet
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

#Flattening
cnn.add(tf.keras.layers.Flatten())

#Full connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

#Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

print("Convnet built")

#Compiling the CNN
cnn.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#Training the CNN
cnn.fit(x = training_set, validation_data = test_set, epochs=30)



test_image = image.load_img('C:\\Users\\lucas\\Desktop\\Deep Learning\\Deep Learning A-Z\\Machine Learning A-Z (Codes and Datasets)\\Part 8 - Deep Learning\\Section 40 - Convolutional Neural Networks (CNN)\\Python\\Section 40 - Convolutional Neural Networks (CNN)\\dataset\\single_prediction\\cat_or_dog_1.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis=0) #adding dimension corresponding to a batch


res = cnn.predict(test_image/255.0) #it's a batch
training_set.class_indices #to know whether dog is 0 or 1

#access to the single element of the batch
if res[0][0] > 0.5:
    print("It's a dog")

else:
    print("It's a cat ")
