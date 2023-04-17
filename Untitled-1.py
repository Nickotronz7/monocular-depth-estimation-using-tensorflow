# %%

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

# %%

input_shape = (256, 256, 3)
kerSize = 3


# %%
model = Sequential([
    # Capas convolucionales
    layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=3, activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, kernel_size=3, activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(256, kernel_size=3, activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Capa de nivelacion
    layers.Flatten(),

    # Capa de conexion total
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='linear')
])

# %%
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error')

# %%
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "/mnt/s/Proyects/dataset/train/outdoor",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="training"
)

# %%
val_data = tf.keras.preprocessing.image_dataset_from_directory(
    "/mnt/s/Proyects/dataset/val",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="validation"
)

# %%
model.fit(train_data, epochs=10, validation_data=val_data)

# %%
import numpy as np
from PIL import Image

# cargar la imagen
image = Image.open("/mnt/s/Proyects/dataset/val/outdoor/scene_00022/scan_00193/00022_00193_outdoor_000_020.png")
image = np.array(image)
image = tf.image.resize(image, (256, 256))
image = np.expand_dims(image, axis=0)

# realizar la prediccion
prediction = model.predict(image)

# Print the predicted depth value
print("Predicted depth value:", prediction[0][0])

# %%
model.save('model2.h5')


