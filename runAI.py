# Code that runs on raspberry
import memory_profiler
import tflite_runtime as lite
from tflite_runtime.interpreter import Interpreter
import numpy as np
from PIL import Image
import time

@memory_profiler.profile
def runAI():
    # load the model
    interpreter = Interpreter(model_path="/home/nicko/tfg_nick/my_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # print(output_details[0]['shape'])

    image = Image.open("/home/nicko/tfg_nick/testpic.png")
    image = image.resize((256,256))
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], image)
    
    #realiza la prediccion
    start = time.time()
    interpreter.invoke()
    end = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(str(output_data[0][0]) + ' took ' + str(end-start))

if __name__=='__main__':
    runAI()