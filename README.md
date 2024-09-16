# A3C for Kung Fu

This project implements the A3C (Asynchronous Advantage Actor-Critic) algorithm to train an AI agent to play the "Kung Fu Master" game using the Gymnasium environment. The model uses deep reinforcement learning techniques with PyTorch and OpenCV for image processing.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Training the Model](#training-the-model)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Installation

1. Install the required packages:
    ```bash
    pip install gymnasium
    pip install "gymnasium[atari, accept-rom-license]"
    pip install gymnasium[box2d]
    pip install torch torchvision
    pip install opencv-python
    ```

2. (Optional) If you use Google Colab, ensure to install the dependencies as shown in the code:
    ```python
    !pip install gymnasium
    !pip install "gymnasium[atari, accept-rom-license]"
    !apt-get install -y swig
    !pip install gymnasium[box2d]
    ```

## Usage

1. Import the necessary libraries and modules in your Python script or Jupyter Notebook.
2. Build the AI model using the provided architecture in the `Network` class.
3. Train the model on the "Kung Fu Master" game using the `Agent` class.
4. Evaluate the model's performance using the `evaluate` function.

## Project Structure

- `a3c_kungfu.ipynb`: The main notebook containing the code for the A3C implementation, model architecture, training loop, and evaluation.
- `utils/`: Utility functions for data preprocessing, model evaluation, etc.
- `README.md`: Overview and instructions for the project.

## Model Details

The A3C algorithm used in this project leverages a convolutional neural network (CNN) to process the game screen frames. The network consists of:
- Three convolutional layers for extracting spatial features.
- A fully connected layer to process the extracted features.
- Separate layers for estimating action values and state values.

## Training the Model

The model is trained using multiple environments simultaneously, using the following hyperparameters:
- Learning rate: `1e-4`
- Discount factor: `0.99`
- Number of environments: `10`

To start training, initialize the `Agent` class and run the training loop:

agent = Agent(number_actions)
# Training loop here

Results
During training, the agent's performance improves gradually. The average reward across episodes is logged to monitor progress.

Example output:
Average agent reward: 440.0
Average agent reward: 1350.0

Acknowledgments
Gymnasium - For providing the environment to train the AI agent.
PyTorch - For the neural network implementation.
OpenAI - For research inspiration in reinforcement learning.
