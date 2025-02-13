# Federated Learning for IoT Intrusion Detection

Description here

## Requirements

* Python 3 installed
* Bash for Linux or Git Bash for Windows
* This repo cloned

## Environment Setup

To setup your environment, use the following command from the project root directory:

```bash
$ bash setup.sh
```

That will create a venv environment and setup all dependencies.

## Running the Simulation

To run the simulation with default arguments, simply run:

```bash
$ flwr run .
```

Configuration

Note: To configure the project, see pyproject.toml.

## TensorBoard

To see results with TensorBoard, run the following:

```bash
$ python env/Lib/site-packages/tensorboard/main.py --logdir="lightning_logs"
```
