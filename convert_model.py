import tensorflow as tf

# Load the Keras model
keras_model = tf.keras.models.load_model("mobilenet.h5")

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open("mobilenet.tflite", "wb") as f:
    f.write(tflite_model)
