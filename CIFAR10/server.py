import flwr as fl
import os

fl.server.start_server(config={"num_rounds": 3})