import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, ReLU, Add, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import os

tf.debugging.set_log_device_placement(True)

tf.keras.backend.clear_session()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show errors

# Set parameters
batch_size = 4
img_height = 224
img_width = 224

# Load training data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r'/home/prasanga-uprety/Documents/resnet/Training',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Load testing data
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r'/home/prasanga-uprety/Documents/resnet/Testing',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Check the class names
class_names = train_ds.class_names
print("Class names:", class_names)
def resnet_block(x, filters, kernel_size=(3, 3), strides=1):
    shortcut = x  # Store the input for the shortcut connection
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)

    # Match the dimensions of the shortcut and output
    if x.shape[-1] != shortcut.shape[-1]:  # If the number of filters is different
        shortcut = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])  # Add the shortcut to the output
    x = ReLU()(x)  # Activation after addition
    return x

def resnet_block_with_projection(x, filters, kernel_size=(3, 3), strides=1):
    shortcut = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(x)  # Adjust shortcut
    shortcut = BatchNormalization()(shortcut)

    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut])  # Add the main path to the shortcut
    x = ReLU()(x)  # Activation after addition
    return x

# Inspect a batch to get the shape of images
images, labels = next(iter(train_ds))
input_shape = images.shape[1:]  # Exclude the batch size
print("Input shape of images:", input_shape)  # This will print (img_height, img_width, channels)

def resnet_build(input_shape=(224, 224, 3), num_classes=4):
    inputs = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(2, 2), strides=1, padding='same')(inputs)
    x = ReLU()(x)

    # First two blocks with 64 filters
    x = resnet_block(x, filters=64)
    x = resnet_block(x, filters=64)

    # Two blocks with projection to 128 filters
    x = resnet_block_with_projection(x, filters=128)
    x = resnet_block_with_projection(x, filters=128)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model


from tensorflow.keras.models import load_model
model = load_model("accuracy_87.h5")
resnet_model = resnet_build()
resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
resnet_model.summary()

# Compile the model before fitting
resnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history = resnet_model.fit(train_ds, validation_data=test_ds, epochs=15)

resnet_model.save("accuracy_87.h5")

tf.debugging.set_log_device_placement(True)



#now lets fit the data
pred = model.predict(test_ds)
prediction = np.argmax(pred , axis = 1)
print(prediction)

true_labels = np.concatenate([y for x, y in test_ds], axis=0)

# Print true labels
print("True labels:", true_labels)

# Calculate prediction accuracy
correct_predictions = np.sum(prediction == true_labels)
accuracy = correct_predictions / len(true_labels)
print(f"Prediction accuracy: {accuracy * 100:.2f}%")

loss, accuracy = model.evaluate(train_ds)
print(f'Training Accuracy: {accuracy * 100:.2f}%')



loss , accuracya = model.evaluate(test_ds)
print(f'Training Accuracy: {accuracya * 100:.2f}%')

#now for prediction
predictions = model.predict(test_ds)
predictions_label = np.argmax(predictions , axis =1)

print(predictions_label)

# Initialize empty lists to store images and labels
true_labels = []

# Iterate through the test dataset
for images, labels in test_ds:
    true_labels.extend(labels.numpy())  # Convert tensor labels to numpy and add to list

# Convert list to numpy array
true_labels = np.array(true_labels)

#accuracy
from sklearn.metrics import accuracy_score
print(f'Accuracy : {accuracy_score(predictions_label , true_labels)*100}')



print("Predicted labels:", predictions_label[:10])
print("True labels:", true_labels[:10])
