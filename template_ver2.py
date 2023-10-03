import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
from keras import layers
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

matplotlib.use('TkAgg')
# Got this code from stackoverflow
# https://stackoverflow.com/questions/73745245/error-using-matplotlib-in-pycharm-has-no-attribute-figurecanvas
# resolves AttributeError: module 'backend_interagg' has no attribute 'FigureCanvas'

###########################MAGIC HAPPENS HERE##########################
# Change the hyper-parameters to get the model performs well
# looks done
config = {
    'batch_size': 64,
    'image_size': (30, 30),
    'epochs': 3,
    'optimizer': keras.optimizers.experimental.SGD(1e-2)
}


###########################MAGIC ENDS  HERE##########################

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
            ###########################MAGIC HAPPENS HERE##########################
            # Use dataset augmentation methods to prevent overfitting
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.3)
            ###########################MAGIC ENDS HERE##########################
        ]
    )
    ds = ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(input_shape, num_classes):

    # hidden_units = 4
    # inputs = keras.Input(shape=input_shape)
    # x = layers.Rescaling(1. / 255)(inputs)
    # x = layers.Flatten()(x)
    # x = layers.Dense(hidden_units, activation='relu')(x)

    ###########################MAGIC HAPPENS HERE##########################
    # Build up a neural network to achieve better performance.
    # Use Keras API like `x = layers.XXX()(x)`
    # Hint: Use a Deeper network (i.e., more hidden layers, different type of layers)
    # and different combination of activation function to achieve better result.
    train_ds, val_ds, test_ds = read_data()
    x_input = keras.Input(shape=input_shape)

    # Add layers to your model using the functional API
    x = layers.Dense(64, activation='relu')(x_input)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense((32 / 2), activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x_output = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=x_input, outputs=x_output)

    model.compile(optimizer='SaucyBoy',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    features_train = np.concatenate([x for x, y in test_ds], axis=0)
    label_train = np.concatenate([y for x, y in test_ds], axis=0)
    prediction_test = np.argmax(model.predict(test_images), 1)

    # validation_data is a tuple containing the validation data features and labels
    # validation_data=(X_val, y_val)  -- unsure if needed or if defined when configured
    model.fit(features_train, label_train)

    # this is just testing
    test_loss, test_accuracy = model.evaluate(prediction_test, test_labels)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    model.save('SaucyBoy.h5')  # saving the model

    ###########################MAGIC ENDS HERE##########################
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
    ###########################MAGIC HAPPENS HERE##########################
    print(history.history)
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print("\nTest Accuracy: ", test_acc)
    test_images = np.concatenate([x for x, y in test_ds], axis=0)
    test_labels = np.concatenate([y for x, y in test_ds], axis=0)
    test_prediction = np.argmax(model.predict(test_images), 1)

    # 1. Visualize the confusion matrix by matplotlib and sklearn based on test_prediction and test_labels

    # # print("==========================")
    # print(test_images)
    # print("\n")
    # print(test_labels)
    # print("\n")
    # print(test_prediction)
    # # print("==========================")

    # # create and display a confusion matrix
    # cm = confusion_matrix(test_labels, test_prediction)
    # cm_display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    # cm_display.plot()
    # plt.show()

    # 2. Report the precision and recall for 5 different classes
    # Hint: check the precision and recall functions from sklearn package or you can implement these function by yourselves.
    print("==========================")
    precision_per_class = precision_score(test_labels, test_prediction, average=None)
    print(precision_per_class)
    print("==========================")

    # 3. Visualize three misclassified images
    # Hint: Use the test_images array to generate the misclassified images using matplotlib
    misclassifiedIndexes = []
    for i in range(len(test_prediction)):
        if test_prediction[i] != test_labels[i]:
            misclassifiedIndexes.append(i)
        if len(misclassifiedIndexes) > 3:
            break

    misclassifiedImages = [test_images[misclassifiedIndexes[0]], test_images[misclassifiedIndexes[1]],
                           test_images[misclassifiedIndexes[2]]]
    print(misclassifiedImages[0])
    print(type(misclassifiedImages))
    for image in misclassifiedImages:
        pass

    print("----------------------------------------------")
    print(misclassifiedIndexes)

    print("ALL DONE")

    ###########################MAGIC ENDS HERE##########################
