# Federated Learning for IoT Intrusion Detection

Description here

## Requirements

* Python 3 installed
* Bash for Linux or Git Bash for Windows
* This repo cloned

## Environment Setup

To setup your environment, use the following command from the project root directory:

```bash
bash setup.sh
```

That will create a venv environment and setup all dependencies.

## Running the Simulation

To run the federated learning simulation with default arguments, simply run:

```bash
flwr run .
```

To run the single worker experiment (non-federated), run:

```
python fliir/single_worker_app.py
```

## Configuration

Note: To configure the project, see pyproject.toml.

## TensorBoard

To see results with TensorBoard, run the following:

```bash
bash run_tensorboard.py
```
