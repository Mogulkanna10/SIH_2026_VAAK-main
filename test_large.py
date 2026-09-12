import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="pi_deploy/dracarys_kws.tflite")
interpreter.allocate_tensors()
output_details = interpreter.get_output_details()
print("Large model output shape:", output_details[0]['shape'])
