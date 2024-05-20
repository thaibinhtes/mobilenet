from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

class_labels =  ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']
print("Class labels:", class_labels)

# Load the Keras model
model = tf.keras.models.load_model('my_model.keras')

# Load the image
img_path = 'test.jpg'  # Replace 'flower.jpg' with the path to your image
img = image.load_img(img_path, target_size=(224, 224))  # Assuming your model expects input size of 224x224
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Normalize pixel values


# Perform inference
predictions = model.predict(img_array)

# Assuming it's a classification model, print the predicted class label
predicted_class = np.argmax(predictions)
predicted_class_label = class_labels[predicted_class]
print("Predicted class label:", predicted_class_label)
