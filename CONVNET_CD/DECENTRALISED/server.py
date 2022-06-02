from typing import Any, Callable, Dict, List, Optional, Tuple
from tensorflow import keras
from tensorflow.keras import layers

import flwr as fl
import tensorflow as tf
import numpy as np

def load_dataset_test():
    num_classes = 10
    _, (x_test, y_test) = keras.datasets.mnist.load_data()

    #Scale images to the [0, 1] range
    x_test = x_test.astype("float32") / 255

    x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print('Dataset loaded on the server side.')
    return x_test, y_test

def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
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

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_eval=0.3,
        min_fit_clients=4,
        min_eval_clients=4,
        min_available_clients=8,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": 10},
        strategy=strategy,
    )

    model.save("convnet_federated")


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    x_val, y_val = load_dataset_test() #len(x_train or y_train) == 10 000
    print("evaluation datsets loaded, x_val: {x_val}, y_val: {y_val}.".format(x_val = x_val.shape, y_val=y_val.shape))

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 24,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()