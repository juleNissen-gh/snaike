"""
Snake AI Training Module

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
Date: 30/7/24
"""

import cProfile
import cloudpickle
import colorama
import numpy as np
import os
import psutil
import signal
import sys
import torch as pt
import torch.multiprocessing as mp
import traceback
import warnings
from torch import nn
from multiprocessing import Queue

from utils.game import Environment
from utils.graph import Graph
from utils.model import Model, optimize
from utils.per import PrioritizedReplayMemory, Staticmem
from utils.play import play

colorama.init()

device = pt.device('cuda')
pt.set_default_device(device)

warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.distributed_c10d")
# <editor-fold desc="Definitions... ">

# GAMMA is the discount factor
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# LR is the learning rate
# BATCH_SIZE is the number of transitions sampled from the replay buffer
TAU = 0.03
epsilon = 1
EPS_END = 0
EPS_DECAY = 0.997
LR = 4e-4
NUM_MEMORY_ITEMS = 30_000
BATCH_SIZE = 2000
UPDATE_FREQUENCY = 1
BG_PROCESSES = 1

TRUNCATED_MATRIX_SIZE = 15
Environment.TRUNCATED_MATRIX_SIZE = TRUNCATED_MATRIX_SIZE
CONV2_TRUNCATED_MATRIX_SIZE = 9

PER_KWARGS = {
    'mem_len': NUM_MEMORY_ITEMS,
    'input_len': Environment.LEN_INPUTS,
    'truncated_matrix_size': Environment.TRUNCATED_MATRIX_SIZE,
    'alpha': 0.75,
    'beta': 0.4,
    'beta_increment': 0.0001,
    'beta_end': 0.7,
    'device': device
}

LOSS_AVG_LEN = 20
SCORE_AVG_LEN = 200
SCORE_WEIGHT_SD = SCORE_AVG_LEN / 3

BOARD_SIZE = 15
ZOOM = 40

GAMMA = 0.94
Environment.LIVING_PENALTY = -0.2
Environment.FOOD_REWARD =15
Environment.DEATH_PENALTY = -10
Environment.WRONG_TURN_PENALTY = Environment.DEATH_PENALTY
Environment.BORDER_PENALTY = -0.25


speed_mod = 3
paused = False
step = False

env_init_kwargs = {
    'zoom': ZOOM,
    'board_size': BOARD_SIZE,
    'truncated_matrix_size': TRUNCATED_MATRIX_SIZE
}

UP = "\x1B[8A"
CLR = "\x1B[0K"


# </editor-fold>

def terminate_process(pid: int, process: mp.Process, terminate_event: mp.Event) -> None:
    """
    Safely terminate a given process.

    Args:
        pid (int): Process ID
        process (multiprocessing.Process): Process object to terminate
        terminate_event (multiprocessing.Event): Event to signal termination

    This function attempts to terminate the process gracefully, and if unsuccessful,
    forcibly terminates it.
    """

    terminate_event.set()
    process.join(timeout=None)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            os.kill(pid, signal.SIGKILL)


def set_high_priority() -> None:
    """
    Set the current process priority to high (Windows only).

    This function is used to ensure consistent performance during training.
    """

    psutil.Process(os.getpid()).nice(psutil.HIGH_PRIORITY_CLASS)


def fill_queue(model: nn.Module, queue: Queue) -> None:
    """
    Fill the given queue with the model's state dict and current epsilon value.

    Args:
        model (nn.Module): The neural network model
        queue (multiprocessing.Queue): Queue to fill with model data

    This function is used to update the model state in child processes.
    """

    model.cpu()
    while not queue.full():
        queue.put((model.state_dict(), epsilon))
    model.to(device=device)


def main() -> None:
    """
    Main training loop for the Snake AI.

    This function sets up the training environment, initializes the neural network models,
    manages the multi-processing for parallel experience sampling, and runs the main training loop.
    It also handles user interactions for saving models, adding/removing processes, and
    visualizing training progress.

    The training process involves:
    1. Collecting gameplay experiences from multiple processes
    2. Updating the replay memory with new experiences
    3. Periodically training the neural network on batches from the replay memory
    4. Updating the networks
    5. Adjusting the exploration rate (epsilon)
    6. Visualizing training metrics

    The function runs indefinitely until manually stopped or an 'end program' signal is received.
    """

    global epsilon

    manager = mp.Manager()

    s = Environment.LIVING_PENALTY
    print('s =', Environment.DEATH_PENALTY + Environment.LIVING_PENALTY + Environment.BORDER_PENALTY,
          round([s := Environment.LIVING_PENALTY + s * GAMMA for _ in range(100)][-1], 3))

    process_dict = {}

    def spawn_process(verbose_args: dict, is_verbose: bool = False) -> int:

        """
        Spawn a new process for parallel gameplay and data collection.

        Args:
            verbose_args (dict): A dictionary containing arguments for the gameplay process,
                                 including settings for visualization and debugging.
            is_verbose (bool, optional): Flag to indicate if this is a verbose process
                                         (i.e., with visualization). Defaults to False.

        Returns:
            int: The process ID (pid) of the spawned process.

        This function creates a new process that runs the 'play' function, which simulates
        gameplay and collects experience data. It uses cloudpickle to serialize the 'play'
        function, allowing it to be run in a separate process.

        The spawned process is added to the process_dict for management and potential
        termination later. If the process is marked as verbose, it will run with visualization
        enabled for debugging and observation purposes.

        Raises:
            Exception: If there's an error in spawning the process, it prints the error
                       and traceback, then re-raises the exception.
        """

        try:
            terminate_event = manager.Event()

            serialized_play = cloudpickle.dumps(play)

            process = mp.Process(target=cloudpickle.loads(serialized_play),
                                 args=(data_q, model_q, model_params, verbose_args, terminate_event, device,
                                       env_init_kwargs))
            process.start()
            process_dict[process.pid] = (process, terminate_event, is_verbose)
            return process.pid
        except Exception as e:
            print(f"Error spawning process: {e}")
            print("Traceback:")
            traceback.print_exc()
            raise

    # <editor-fold desc="vars...">
    plot = Graph(UPDATE_FREQUENCY, SCORE_WEIGHT_SD)
    set_high_priority()
    model_params = (Environment.LEN_INPUTS, 60, 4, TRUNCATED_MATRIX_SIZE, CONV2_TRUNCATED_MATRIX_SIZE)
    average_loss = np.array([])
    scores = np.array([], dtype=np.uint8)
    policy_net = Model(*model_params).float()
    target_net = Model(*model_params).to(device=device)
    grok_grads = None

    # state_dict = pt.load('models\\modell88.pt')
    # epsilon = EPS_END
    # PER_KWARGS['beta'] = PER_KWARGS['beta_end']
    # policy_net.load_state_dict(state_dict)

    target_net.load_state_dict(policy_net.state_dict())
    replay_memory = PrioritizedReplayMemory(**PER_KWARGS)
    criterion = nn.MSELoss(reduction='none')
    optimizer = pt.optim.Adam(policy_net.parameters(), lr=LR)
    episode = 0
    data_q = manager.Queue()
    model_q = manager.Queue(maxsize=UPDATE_FREQUENCY + 5)
    high_score = 0
    verbose_args = {
        'verbose': 1,
        'speed_mod': 4,
        'paused': 0,
        'step': 0,
        'n_steps': float('inf')
    }
    non_verbose_args = {k: 0 for k, _ in verbose_args.items()}
    non_verbose_args['n_steps'] = 100

    for _ in range(BG_PROCESSES):
        spawn_process(non_verbose_args)
    spawn_process(verbose_args, is_verbose=True)

    fill_queue(policy_net, model_q)

    # </editor-fold>

    pt.set_default_device(device)

    while True:
        membatch = data_q.get()

        if membatch[0] == -1:  # message to save or terminate
            if membatch[1] == 'save':
                pt.save(policy_net.state_dict(), f'models/modell{high_score}.pt')

            elif membatch[1] == 'end program':
                # Terminate all processes
                for pid, (process, terminate_event, _) in process_dict.items():
                    terminate_process(pid, process, terminate_event)
                plot.terminate()
                break

            elif membatch[1] == 'rmprocess':
                if process_dict:
                    # Remove a non-verbose process
                    for pid, (process, terminate_event, is_verbose) in list(process_dict.items()):
                        if not is_verbose:
                            terminate_process(pid, process, terminate_event)
                            del process_dict[pid]
                            break

            elif membatch[1] == 'process':
                spawn_process(non_verbose_args)

            continue

        high_score = max(high_score, membatch[1])
        scores = np.append(scores, membatch[1])
        replay_memory.push(Staticmem(*[i.to(device=device) for i in membatch[0]]))
        del membatch

        epsilon = EPS_END + (epsilon - EPS_END) * EPS_DECAY

        if len(scores) > SCORE_AVG_LEN:
            scores = scores[1:]

        # Training loop
        if episode % UPDATE_FREQUENCY == 0:  # update every UPDATE_FREQ game
            fill_queue(policy_net, model_q)  # update simulators' models (and epsilon)

            loss, grok_grads = optimize(policy_net=policy_net, target_net=target_net, memory=replay_memory,
                                        criterion=criterion, optimizer=optimizer, batch_size=BATCH_SIZE,
                                        gamma=GAMMA, tau=TAU, grok_grads=grok_grads)

            average_loss = np.append(average_loss, loss)

            if len(average_loss) > LOSS_AVG_LEN:
                average_loss = average_loss[1:]

            print(f"""{UP}Episode: {episode:>8}{CLR}
Loss: {np.mean(average_loss):>11.2f}{CLR}
Epsilon: {epsilon:>8.1%}{CLR}
High Score: {high_score:>5}{CLR}
Avg. Score: {np.mean(scores):>5.1f}{CLR}
Dataqs: {data_q.qsize():>9}{CLR}
Processes: {len(process_dict.items()):>6}{CLR}
Beta: {replay_memory.beta:>11.3f}{CLR}""")

            plot.update_plot(loss, average_loss, scores)

        episode += 1
        plot.event_loop()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    if '--profile' in sys.argv:
        mp.freeze_support()
        BG_PROCESSES = 5
        profiler = cProfile.Profile()
        profiler.enable()
        main()
        profiler.disable()
        print('Saving profiling data')
        profiler.dump_stats('output.prof')
    else:
        main()
