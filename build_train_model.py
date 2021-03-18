import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

training_dataset = "/home/agrivision/detection/Dataset/train"
validation_dataset = "/home/agrivision/detection/Dataset/validation"


# trying to generate images from differ angles 
training_generator = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
# saving generated images in a variable 
tr_generated_imgs = training_generator.flow_from_directory(training_dataset,
                                       target_size=(150, 150),
                                       batch_size=32,
                                       class_mode='binary')
#just for check image classes we are using in training
tr_generated_imgs.class_indices


#////////////////////////////////////////////////////////////////////////////

#just rescaling for validation
validation_generator = ImageDataGenerator(rescale=1./255)

#saving validation generated images that we rescaled 
vl_generated_imgs = validation_generator.flow_from_directory(validation_dataset,
                                        target_size=(150, 150),
                                        batch_size=32,
                                        class_mode='binary')
                                                             

# Show images using matplotlib 

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip (images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
    
#  ploting images using PlotImages function
images = [tr_generated_imgs[0][0][0] for i in range(5)]
plotImages(images)


#////////////////////////////////////////////////////////////////////////////////


#Building CNN model

cnn_model = keras.models.Sequential([
                    keras.layers.Conv2D(filters=32, kernel_size=3,input_shape=[150, 150, 3]),
                    keras.layers.MaxPooling2D(pool_size=(2,2)),
                    keras.layers.Conv2D(filters=64, kernel_size=3),
                    keras.layers.MaxPooling2D(pool_size=(2, 2)),
                    keras.layers.Conv2D(filters=128, kernel_size=3),
                    keras.layers.MaxPooling2D(pool_size=(2, 2)),
                    keras.layers.Conv2D(filters=256, kernel_size=3),
                    keras.layers.MaxPooling2D(pool_size=(2, 2)),
                    
                    keras.layers.Dropout(0.5),
                    keras.layers.Flatten(), #neural network building
                    keras.layers.Dense(units=128, activation='relu'), # input layer
                    keras.layers.Dropout(0.1),
                    keras.layers.Dense(units=256, activation='relu'),
                    keras.layers.Dropout(0.25),
                    keras.layers.Dense(units=2, activation='softmax') #outpt layeru
                    ])

#/////////////////////////////////////////////////////////////////////////


#compiling cnn_model

cnn_model.compile(optimizer = Adam(lr=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


# model path for saving best accurate with help of check points

model_path= '/home/agrivision/detection/model/predictor.h5'
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')
callback_list = [checkpoint]

#////////////////////////////////////////////////////////////////////////////


#train our cnn_model

history = cnn_model.fit(tr_generated_imgs,
                        epochs=100,
                        verbose=1,
                        validation_data=vl_generated_imgs,
                        callbacks=callback_list)



#//////////////////////////////////////////////////////////////////////////////

#plot history for accuracy

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#plot history for loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()