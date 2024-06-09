import tensorflow as tf
import matplotlib.pyplot as plt

path_to_train_dataset = r"C:\Users\krzyc\Documents\prac_mag_git\backend\dataset\train"
path_to_test_dataset = r"C:\Users\krzyc\Documents\prac_mag_git\backendk\dataset\test"

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)

train_generator = train_datagen.flow_from_directory(path_to_train_dataset,
                                                    batch_size=100,
                                                    class_mode='binary',
                                                    target_size=(28, 28),
                                                    color_mode='grayscale')

validation_generator = validation_datagen.flow_from_directory(path_to_test_dataset,
                                                              batch_size=100,
                                                              class_mode='binary',
                                                              target_size=(28, 28),
                                                              color_mode='grayscale')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(strides=(2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')  # Ensure num_classes is defined somewhere above, or replace with 10
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit_generator(train_generator, epochs=5, verbose=1,
                              validation_data=validation_generator)


print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Dokładność Modelu')
plt.ylabel('Dokładność')
plt.xlabel('Epoka')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Straty modelu')
plt.ylabel('Strata')
plt.xlabel('Epoka')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
