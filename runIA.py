# Code that runs on rasp adapted to pc
import memory_profiler
import tensorflow as tf
import numpy as np
from PIL import Image
import time


@memory_profiler.profile
def runAI():
    # load the model
    interpreter = tf.lite.Interpreter(
        model_path="/home/nicko/proyects/monocular-depth-estimation-using-tensorflow/my_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = Image.open(
        "/home/nicko/proyects/monocular-depth-estimation-using-tensorflow/pics/10.png")
    image = image.resize((256, 256))
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], image)

    # realiza la prediccion
    start = time.time()
    interpreter.invoke()
    end = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(str(output_data[0][0]) + ' took ' + str(end-start))


if __name__ == '__main__':
    runAI()
