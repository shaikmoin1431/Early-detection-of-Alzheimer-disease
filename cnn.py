import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Dropout, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_addons as tfa
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
 
# Define the data directories
train_dir = r"C:\Users\saiki\OneDrive\Desktop\alze_dataset\alzehmer_dataset_pca\train"
test_dir = r"C:\Users\saiki\OneDrive\Desktop\alze_dataset\alzehmer_dataset_pca\test"

# Define the image size and batch size
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# Define the classes
CLASSES = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']

# Create data generators
train_data_gen = ImageDataGenerator(rescale=1.0/255.0)
test_data_gen = ImageDataGenerator(rescale=1.0/255.0)

train_data = train_data_gen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_data_gen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

def conv_block(filters, act='relu'):
    """Defining a Convolutional NN block for a Sequential CNN model. """
    
    block = Sequential()
    block.add(Conv2D(filters, 3, activation=act, padding='same'))
    block.add(Conv2D(filters, 3, activation=act, padding='same'))
    block.add(BatchNormalization())
    block.add(MaxPool2D())
    
    return block

def dense_block(units, dropout_rate, act='relu'):
    """Defining a Dense NN block for a Sequential CNN model. """
    
    block = Sequential()
    block.add(Dense(units, activation=act))
    block.add(BatchNormalization())
    block.add(Dropout(dropout_rate))
    
    return block

def construct_model(act='relu'):
    """Constructing a Sequential CNN architecture for performing the classification task. """
    
    model = Sequential([
        Input(shape=(*IMAGE_SIZE, 3)),
        Conv2D(16, 3, activation=act, padding='same'),
        Conv2D(16, 3, activation=act, padding='same'),
        MaxPool2D(),
        conv_block(32),
        conv_block(64),
        conv_block(128),
        Dropout(0.2),
        conv_block(256),
        Dropout(0.2),
        Flatten(),
        dense_block(512, 0.7),
        dense_block(128, 0.5),
        dense_block(64, 0.3),
        Dense(len(CLASSES), activation='softmax')  # Change the output units to match the number of classes
    ], name="cnn_model")

    return model

# Defining a custom callback function to stop training our model when accuracy goes above 99%
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_acc') > 0.99:
            print("\nReached accuracy threshold! Terminating training.")
            self.model.stop_training = True

my_callback = MyCallback()

# EarlyStopping callback to make sure the model is always learning
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# Defining other parameters for our CNN model
model = construct_model()

METRICS = [tf.keras.metrics.CategoricalAccuracy(name='acc'),
           tf.keras.metrics.AUC(name='auc'), 
           tfa.metrics.F1Score(num_classes=4)]

CALLBACKS = [my_callback]

model.compile(optimizer='adam',
              loss=tf.losses.CategoricalCrossentropy(),
              metrics=METRICS)

model.summary()

# Fit the training data to the model and validate it using the validation data
EPOCHS = 25

history = model.fit(train_data, validation_data=test_data, callbacks=CALLBACKS, epochs=EPOCHS)

# Plotting the trend of the metrics during training
fig, ax = plt.subplots(1, 3, figsize=(30, 5))
ax = ax.ravel()

for i, metric in enumerate(["acc", "auc", "loss"]):
    ax[i].plot(history.history[metric])
    ax[i].plot(history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("Epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

# Evaluating the model on the test data
test_scores = model.evaluate(test_data)

print("Testing Accuracy: %.2f%%" % (test_scores[1] * 100))

# Predicting the test data
pred_labels = model.predict(test_data)

# Convert softmax predictions to binary labels
def roundoff(arr):
    arr[np.argwhere(arr != arr.max())] = 0
    arr[np.argwhere(arr == arr.max())] = 1
    return arr

for labels in pred_labels:
    labels = roundoff(labels)

# Print the classification report of the test data
test_labels = test_data.classes
print(classification_report(test_labels, np.argmax(pred_labels, axis=1), target_names=CLASSES))

# Plot the confusion matrix
pred_ls = np.argmax(pred_labels, axis=1)

conf_arr = confusion_matrix(test_labels, pred_ls)

plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
ax = sns.heatmap(conf_arr, cmap='Greens', annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Alzheimer's Disease Diagnosis")
plt.xlabel("Prediction")
plt.ylabel("Truth")
plt.show()

# Printing some other classification metrics
from sklearn.metrics import balanced_accuracy_score as BAS, matthews_corrcoef as MCC

print("Balanced Accuracy Score: {} %".format(round(BAS(test_labels, pred_ls) * 100, 2)))
print("Matthew's Correlation Coefficient: {} %".format(round(MCC(test_labels, pred_ls) * 100, 2)))

# Saving the model for future use
model_dir = "alzheimer_cnn_model"
model.save(model_dir, save_format='h5')

# Load the saved model
pretrained_model = tf.keras.models.load_model(model_dir)

# Check its architecture
tf.keras.utils.plot_model(pretrained_model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)
