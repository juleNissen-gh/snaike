# snaike
This module implements a reinforcement learning approach to train an AI agent to play the game of Snake.
It uses a deep Q-network (DQN) with prioritized experience replay and multi-processing for parallel experience sampling.

The main components of this module include:
1. Environment setup for the Snake game
2. Neural network model definition
3. Training loop with experience replay
4. Multi-processing for parallel experience sampling
5. Visualization of training progress

Usage:
Run this script directly to start the training process. Use command-line argument '--profile'
for profiling the performance.

Dependencies:
- PyTorch
- Pygame
- NumPy
- Cloudpickle
- Psutil
- Colorama

Author: Philip Nissen-Lie Trøen
Date: 1/1/25
