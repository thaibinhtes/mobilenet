import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load the pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification layers on top of MobileNetV2
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)  # Adjust num_classes to match your dataset

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Optionally freeze some layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Define loss function and optimizer
loss_function = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Compile the model
model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

# # Prepare and load your dataset
# train_dataset = ...
# validation_dataset = ...

# # Fine-tuning loop
# epochs = 10
# for epoch in range(epochs):
#     for images, labels in train_dataset:
#         with tf.GradientTape() as tape:
#             predictions = model(images, training=True)
#             loss = loss_function(labels, predictions)
#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
#     # Evaluate model on validation dataset
#     validation_loss, validation_accuracy = model.evaluate(validation_dataset)
#     print(f'Epoch {epoch+1}, Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}')
