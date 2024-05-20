import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow_datasets as tfds
  
# Load data
(train_data, val_data), info = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True  # Load data as (image, label) pairs
)

IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

def preprocess_image(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = image / 255.0  # Normalize to [0, 1]
    return image, label

# Apply preprocessing and create batches
train_data = train_data.map(preprocess_image, num_parallel_calls=AUTOTUNE)
train_data = train_data.cache().shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

val_data = val_data.map(preprocess_image, num_parallel_calls=AUTOTUNE)
val_data = val_data.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# Check the data shapes
for image, label in train_data.take(1):
    print(image.shape, label.shape)


# Load the pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# có thể thay đổi các trọng số để tối ưu với tệp data khác hiện tại data này từ của model nên trọng số có độ chính xác cao
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(info.features['label'].num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, epochs=1, validation_data=val_data)

model.save('mobile.tflite')
# model = tf.keras.models.load_model('my_model.keras')


# # Convert the model
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_model = converter.convert()

# # # Save the TFLite model
# # with open('mobilenet.tflite', 'wb') as f:
# #     f.write(tflite_model)

