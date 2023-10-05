import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
from keras import layers
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from sklearn.metrics import precision_score

matplotlib.use('TkAgg')
# Got this code from stackoverflow
# https://stackoverflow.com/questions/73745245/error-using-matplotlib-in-pycharm-has-no-attribute-figurecanvas
# resolves AttributeError: module 'backend_interagg' has no attribute 'FigureCanvas'

########################### MAGIC HAPPENS HERE ##########################
# Change the hyper-parameters to get the model performs well

# only change here and hidden_units that is inside build_mode()
config = {
    'batch_size': 32,  # 128 is high and 16 is low
    'image_size': (128, 128),  # 128 is high and 16 is low
    'epochs': 30,  # 5 - 40  46 0.50, 30 got exactly 0.5000
    'optimizer': keras.optimizers.experimental.Adam(learning_rate=0.001)
}


########################### MAGIC ENDS  HERE ##########################

def read_data():
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        "images/flower_photos",
        validation_split=0.2,
        subset="both",
        seed=42,
        image_size=config['image_size'],
        batch_size=config['batch_size'],
        labels='inferred',
        label_mode='int'
    )
    val_batches = tf.data.experimental.cardinality(val_ds)
    test_ds = val_ds.take(val_batches // 2)
    val_ds = val_ds.skip(val_batches // 2)
    return train_ds, val_ds, test_ds


def data_processing(ds):
    data_augmentation = keras.Sequential(
        [
            ########################### MAGIC HAPPENS HERE ##########################
            # Use dataset augmentation methods to prevent overfitting
            layers.RandomFlip("vertical"),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(factor=0.2),
            layers.RandomContrast(factor=0.2)
            ########################### MAGIC ENDS HERE ##########################
        ]
    )
    ds = ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(input_shape, num_classes):

    ########################### MAGIC HAPPENS HERE ##########################
    # Build up a neural network to achieve better performance.
    # Use Keras API like `x = layers.XXX()(x)`
    # Hint: Use a Deeper network (i.e., more hidden layers, different type of layers)
    # and different combination of activation function to achieve better result.

    hidden_units = 64  # 125 - 400
    inputs = keras.Input(shape=input_shape)

    # Add layers to your model using the functional API
    x = layers.Rescaling(1. / 255)(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(hidden_units, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.Dropout(0.3)(x)  # Increased dropout rate
    x = keras.layers.BatchNormalization()(x)  # Batch normalization
    x = keras.layers.Dense(hidden_units // 2, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.Dropout(0.3)(x)  # Increased dropout rate
    x = keras.layers.BatchNormalization()(x)  # Batch normalization

    ########################### MAGIC ENDS HERE ##########################
    outputs = layers.Dense(num_classes, activation="softmax",
                           kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    model = keras.Model(inputs, outputs)
    print(model.summary())
    return model


if __name__ == '__main__':
    # Load and Process the dataset
    train_ds, val_ds, test_ds = read_data()
    train_ds = data_processing(train_ds)
    # Build up the ANN model
    model = build_model(config['image_size'] + (3,), 5)
    # Compile the model with optimizer and loss function
    model.compile(
        optimizer=config['optimizer'],
        loss='SparseCategoricalCrossentropy',
        metrics=["accuracy"],
    )
    # Fit the model with training dataset
    history = model.fit(
        train_ds,
        epochs=config['epochs'],
        validation_data=val_ds
    )
    ########################### MAGIC HAPPENS HERE ##########################
    print(history.history)
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print("\nTest Accuracy: ", test_acc)
    test_images = np.concatenate([x for x, y in test_ds], axis=0)
    test_labels = np.concatenate([y for x, y in test_ds], axis=0)
    test_predictions = np.argmax(model.predict(test_images), 1)

    # 1. Visualize the confusion matrix by matplotlib and sklearn based on test_prediction and test_labels

    # # print("==========================")
    # print(test_images)
    # print("\n")
    # print(test_labels)
    # print("\n")
    # print(test_prediction)
    # # print("==========================")

    # # create and display a confusion matrix
    cm = confusion_matrix(test_labels, test_predictions)
    # cm_display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    # cm_display.plot()
    # plt.show()
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


    # 2. Report the precision and recall for 5 different classes Hint: check the precision and recall functions from
    # sklearn package, or you can implement these function by yourselves.
    # print(cm)
    print("..................")

    precision = precision_score(test_labels, test_predictions, average=None)
    recall = recall_score(test_labels, test_predictions, average=None)
    for i in range(len(precision)):
        print(f"Flower {i} - Precision: {precision[i]}, Recall: {recall[i]}")

    macro_precision = precision_score(test_labels, test_predictions, average='macro')
    macro_recall = recall_score(test_labels, test_predictions, average='macro')
    macro_f1 = f1_score(test_labels, test_predictions, average='macro')

    weighted_precision = precision_score(test_labels, test_predictions, average='weighted')
    weighted_recall = recall_score(test_labels, test_predictions, average='weighted')
    weighted_f1 = f1_score(test_labels, test_predictions, average='weighted')

    # 3. Visualize three misclassified images
    # Hint: Use the test_images array to generate the misclassified images using matplotlib
    misclassified_indexes = np.where(test_labels != test_predictions)[0][:3]
    misclassified_images = test_images[misclassified_indexes]
    for i, image in enumerate(misclassified_images):
        plt.subplot(1, 3, i+1)
        plt.imshow(image)
        plt.title(f"True: {test_labels[misclassified_indexes[i]]}, "
                  f"Predicted: {test_predictions[misclassified_indexes[i]]}")
    plt.show()


    print("----------------------------------------------")
    print("---------SUMMERY----------")
    print("\nTest Accuracy: ", test_acc)
    print("Average Precision: " + str(sum(precision) / len(precision)))
    print("Average Recall: " + str(sum(recall) / len(recall)))

    print("\nALL DONE")

    ########################### MAGIC ENDS HERE ##########################

    ########################### * ABRACADABRA * ##########################
