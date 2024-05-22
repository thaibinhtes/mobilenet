import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow_datasets as tfds

class Retrain:
    def __init__(self, data):
        self.train_data = None
        self.val_data = None
        self.info = None
        self.IMG_SIZE = 224
        self.BATCH_SIZE = 32
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.data = data

    def load_data(self):
        (train_data, val_data), info = tfds.load(
            'tf_flowers',
            split=['train[:80%]', 'train[80%:]'],
            with_info=True,
            as_supervised=True  # Load data as (image, label) pairs
        )

        self.train_data = train_data
        self.val_data = val_data
        self.info = info

    def preprocess_image(self, image, label):
        image = tf.image.resize(image, [self.IMG_SIZE, self.IMG_SIZE])
        image = image / 255.0  # Normalize to [0, 1]
        return image, label

    def create_data(self):
        self.train_data = self.train_data.map(self.preprocess_image, num_parallel_calls=self.AUTOTUNE)
        self.train_data = self.train_data.cache().shuffle(buffer_size=1000).batch(self.BATCH_SIZE).prefetch(buffer_size=self.AUTOTUNE)

        self.val_data = self.val_data.map(self.preprocess_image, num_parallel_calls=self.AUTOTUNE)
        self.val_data = self.val_data.batch(self.BATCH_SIZE).prefetch(buffer_size=self.AUTOTUNE)

    def retrain(self):
        # Load the pre-trained MobileNetV2 model without the top classification layer
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(self.info.features['label'].num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.fit(self.train_data, epochs=1, validation_data=self.val_data)

        model.save('models/model.keras')

    def run(self):
        self.load_data()
        self.create_data()
        self.retrain()
    
    # def convert():
    #   # Load the Keras model
    #   keras_model = tf.keras.models.load_model("models/model.keras")

    #   # Convert the model to TensorFlow Lite format
    #   converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    #   tflite_model = converter.convert()

    #   # Define the filename for the TensorFlow Lite model
    #   filename = "mdoels/model.tflite" # Changed the filename to avoid the error

    #   # Check if the filename is a directory
    #   if os.path.isdir(filename):
    #       raise IsADirectoryError(f"{filename} is a directory")

    #   # Save the TensorFlow Lite model to a file
    #   with open(filename, "wb") as f:
    #       f.write(tflite_model)
