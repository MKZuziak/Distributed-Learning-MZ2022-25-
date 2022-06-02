# MNIST CONVNET - FROM CENTRALIZED TO DECENTRALIZED

#### AUTHOR: Maciej Zuziak
#### CONTACT: maciejkrzysztof.zuziak@isti.cnr.it
#### VERSION: 0.9.1

#### ACKNOWLEDGEMENTS: FLOWER'S ADVANCED TENSORFLOW EXAMPLES (https://github.com/adap/flower/tree/main/examples/advanced_tensorflow)
#### ACKNOWLEDGEMENTS: FCHOLLET'S SIMPLE MNIST CONVNET (https://keras.io/examples/vision/mnist_convnet/)

## DESCRIPTION
A decentralized version of simple MNIST convnet (original version available here: https://keras.io/examples/vision/mnist_convnet/). This simple scenario provides for a training model on no more than 40 agents that each holds a different part of the MNIST dataset. The model is saved by default in the folder #convnet_federated. Code is based on the example provided by FLower's developers (original version available here: https://github.com/adap/flower/tree/main/examples/advanced_tensorflow)

## SIMULATION SETTING:
-  No more than `40 agents` each holding a partition of the original MNIST dataset (by default: 15 agents);
-  By default `10 training rounds`;
-  Batch size: 24 (by default);
-  Local Epochs: 1 or 2 depending on the training round;
-  Fraction fit: 0.3 (each round a 30% of available agents are selected for training);
-  Fraction eval: 0.3 (each round a 30% of available agents are selected for evaluation)
-  min_fit_clients = 4 && min_eval_clients = 4 (server needs to select at least 4 agents for training and evaluation to start)
-  min_available_clients = 8 (at least 8 agents must be connected to the network in order to start operation)
-  gRPC channel insecured

## INSTRUCTIONS:
- It is recommended to deploy this code in a virtual environment. This tutorial use Poetry for dependency management (link: https://python-poetry.org/)
-  After installing Poetry `$pip install --user poetry` navigate to the folder containing `pyproject.toml` files. Type in `poetry install` to prepare the environment and then `poetry shell` to use it. Then you can run `run.sh` to start the simulation (`sh run.sh` on Linux).

## CHANGING THE SIMULATION
- In order to change the simulation, you can either change the `run.sh` file (in order to change the number of clients) or key's values in `fl.server.strategy.FedAvg` dictionary (in `server.py` file). Remember that you have to change the partitioning of the dataset in `load partition function` (in `client.py`) after changing the number of clients.