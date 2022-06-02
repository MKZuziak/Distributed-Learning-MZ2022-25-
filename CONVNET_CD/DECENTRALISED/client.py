import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import flwr as fl

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps, verbose=0)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(0, 40), required=True)
    args = parser.parse_args()
    # Change number of clients in the run.sh file.
    if args.partition > 40:
        raise Exception('The maximal number of clients that may be connected is 40!')

    # Load and compile Keras model
    num_classes = 10
    input_shape = (28, 28, 1)
    model = keras.Sequential([
        keras.Input(shape=input_shape), # Used to instantiate a Keras tensor.
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"), #2D convolution layer.
        layers.MaxPooling2D(pool_size=(2, 2)), # Max pooling operation for 2D spatial data.
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(), #Flatten the input
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Load a subset of CIFAR-10 to simulate the local data partition
    x_train, y_train, x_test, y_test = load_partition(args.partition)

    print('Client {idx} loaded, training set shape: \
        {x_train} and {y_train}, {x_test}, \
        {y_test}'.format(idx = args.partition, x_train=x_train.shape, \
        y_train=y_train.shape, x_test=x_test.shape, y_test=y_test.shape))
    # Start Flower client
    client = CifarClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client("[::]:8080", client=client)
    print('Client {idx} has finished the training - disconnecting'.format(idx = args.partition))


def load_partition(idx: int):
    """Load 1/10 (or 1/i-th) of the training and test data to simulate a partition."""
    
    # Number of classes in the MNIST dataset
    num_classes = 10
   
    # Loading the dataset and asserting proper shape
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    train_fraction = int(len(x_train) / 15) # number of agents
    test_fraction = int(len(x_test) / 15) # number of agents

    x_train = x_train[idx * train_fraction : (idx + 1) * train_fraction]
    y_train = y_train[idx * train_fraction : (idx + 1) * train_fraction]

    x_test = x_test[idx * test_fraction : (idx + 1) * test_fraction]
    y_test = y_test[idx * test_fraction : (idx + 1) * test_fraction]

    #Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    main()
