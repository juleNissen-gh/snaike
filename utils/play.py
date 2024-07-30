"""
This module contains the play function, which is responsible for running the snake game
simulation in a separate process. It interacts with the main training loop by receiving
model updates and sending back experience data for training.

The play function continuously runs game episodes, collects experience data, and
communicates with the main process through multiprocessing queues. It also handles
user input for controlling the visualization and debugging when running in verbose mode.
"""

import gc
import logging
import pygame as pg
import queue
import random
import time
import torch as pt
import torch.multiprocessing as mp

from utils.game import Environment
from utils.model import Model, compute_q_values
from utils.per import Staticmem


def play(data_q: mp.Queue,
         model_q: mp.Queue,
         model_params: tuple[int],
         verbose_args: dict,
         terminate_event: mp.Event,
         device: pt.device,
         env_init_kwargs: dict,
         gamma: float = 0.95) -> None:
    """
    Runs a continuous simulation of the snake game, collecting experience data and
    sending it back to the main process for training.

    This function initializes the game environment, sets up the neural network model,
    and continuously plays episodes of the snake game. It receives model updates from
    the main process and sends back collected experience data. When running in verbose
    mode, it also handles user input for controlling the visualization and debugging.

    Args:
        data_q (mp.Queue): Queue for sending experience data back to the main process.
        model_q (mp.Queue): Queue for receiving model weight updates from the main process.
        model_params (tuple[int]): Parameters for initializing the Model architecture.
        verbose_args (dict): Dictionary containing parameters for verbose mode operation.
        terminate_event (mp.Event): Event to signal termination of the process.
        device (pt.device): PyTorch device to use for tensor operations.
        env_init_kwargs (dict): Keyword arguments for initializing the game environment.
        gamma (float, optional): Discount factor used in Q-value calculations. Defaults to 0.95.

    Returns:
        None

    Note:
        This function runs in an infinite loop until the terminate_event is set.
        It handles exceptions and ensures proper cleanup of CUDA resources before exiting.
    """

    device = pt.device(device)
    pt.set_default_device(device)

    # Warm-up period
    for _ in range(3):
        dummy_data = pt.zeros(1, dtype=pt.float32).cpu()
        data_q.put((-1, dummy_data))
        time.sleep(0.1)  # Short delay

    pt.set_grad_enabled(False)
    Environment.set_board_size(**env_init_kwargs)
    policy_net = Model(*model_params).eval()
    policy_net_dict = None
    epsilon = None
    try:
        while not terminate_event.is_set():
            n_steps = 0

            try:  # get updated model. use previous if timeout
                policy_net_dict, epsilon = model_q.get(timeout=0.1, block=True)
            except queue.Empty:
                if policy_net_dict is None:
                    continue

            policy_net.load_state_dict(policy_net_dict)

            game = Environment(verbose=verbose_args['verbose'],
                               pos=(random.randint(0, Environment.BOARD_SIZE - 3),
                                    random.randint(0, Environment.BOARD_SIZE - 1)))

            # define return data tensors
            rv_states = pt.tensor([], dtype=pt.int8)
            rv_b_matricies = pt.tensor([], dtype=pt.bool)
            rv_next_states = pt.tensor([], dtype=pt.int8)
            rv_next_b_matricies = pt.tensor([], dtype=pt.bool)
            rv_rewards = pt.tensor([], dtype=pt.float16)
            rv_playing = pt.tensor([], dtype=pt.bool)

            prev_states = []

            while game.playing and n_steps < verbose_args['n_steps']:
                if terminate_event.is_set():
                    break
                n_steps += 1

                if verbose_args['verbose']:
                    verbose_args['step'] = False
                    game.clock.tick(1.5 ** verbose_args['speed_mod'])

                    # eventloop
                    for event in pg.event.get():
                        if event.type == pg.QUIT:
                            data_q.put((-1, 'end program'))
                            raise SystemExit

                        elif event.type != pg.KEYDOWN:
                            continue

                        if event.key == pg.K_SPACE:
                            verbose_args['paused'] = not verbose_args['paused']
                            verbose_args['step'] = False
                        elif event.key == pg.K_RETURN:
                            verbose_args['step'] = True
                        elif event.key == pg.K_UP:
                            verbose_args['speed_mod'] += 1
                        elif event.key == pg.K_DOWN:
                            verbose_args['speed_mod'] -= 1
                        elif event.key == pg.K_RIGHT:
                            data_q.put((-1, 'save'))
                        elif event.key == pg.K_PAGEUP or event.key == 1073741921:  # Page Up
                            data_q.put((-1, 'process'))
                        elif event.key == pg.K_PAGEDOWN or event.key == 1073741915:  # Page Down
                            data_q.put((-1, 'rmprocess'))

                    if verbose_args['paused']:
                        if not verbose_args['step']:
                            n_steps -= 1
                            continue

                state, b_matrix = list(game.get_state())  # get state and take step
                state = pt.from_numpy(state).to(device=device, dtype=pt.float)

                # check for the agent looping within the previous 30 steps
                if t := (len(game.snake), game.snake[-1]) in prev_states:
                    break
                prev_states.append(t)
                if len(prev_states) >= 30:
                    prev_states.pop(0)

                b_matrix = pt.from_numpy(b_matrix).to(device=device, dtype=pt.float)

                rewards, next_states, next_b_matricies, playing = (
                    game.step_all(policy_net.select_action((state, b_matrix), epsilon).item()))

                state = state.to(dtype=pt.int8)
                b_matrix = b_matrix.to(dtype=pt.bool)

                # move to pt tensor
                rewards = pt.from_numpy(rewards).to(dtype=pt.float16, device=device)
                next_states = pt.from_numpy(next_states).to(dtype=pt.int8, device=device)
                next_b_matricies = pt.from_numpy(next_b_matricies).to(dtype=pt.bool, device=device)
                playing = pt.from_numpy(playing).to(dtype=pt.bool, device=device)

                # add data to return tensors
                rv_states = pt.cat((rv_states, state.unsqueeze(0)))
                rv_b_matricies = pt.cat((rv_b_matricies, b_matrix.unsqueeze(0)))
                rv_next_states = pt.cat((rv_next_states, next_states.unsqueeze(0)))
                rv_next_b_matricies = pt.cat((rv_next_b_matricies, next_b_matricies.unsqueeze(0)))
                rv_rewards = pt.cat((rv_rewards, rewards.unsqueeze(0)))
                rv_playing = pt.cat((rv_playing, playing.unsqueeze(0)))

                if verbose_args['step']:  # print debug info: q_values
                    print('\n' * 8 * 2, '=' * 10, *[[round(j, 3) for j in i.tolist()] for i in compute_q_values(
                        Staticmem(state.to(device=device, dtype=pt.float),
                                  b_matrix,
                                  next_states.to(device=device, dtype=pt.float),
                                  next_b_matricies,
                                  rewards.to(device),
                                  playing.to(pt.bool),
                                  pt.zeros(1)),

                        policy_net, policy_net, gamma)], '=' * 10, '\n' * 8 * 2, sep='\n', flush=True)

            if verbose_args['verbose']:  # clear screen
                try:
                    game.window.fill(Environment.BLACK)
                except AttributeError:
                    raise Exception('PyGame not initialized')

            data_q.put((Staticmem(
                rv_states.cpu(),
                rv_b_matricies.cpu(),
                rv_next_states.cpu(),
                rv_next_b_matricies.cpu(),
                rv_rewards.cpu(),
                rv_playing.cpu(),
                pt.zeros(rv_states.shape[0]).cpu()),
                        len(game.snake)))

            if Environment.high_score > 10:
                verbose_args['n_steps'] = float('inf')

    except Exception as e:
        logging.error(f"Error in play process: {e}", exc_info=True)

    finally:
        # Ensure all CUDA resources are released
        policy_net.cpu()  # Move model to CPU if it exists

        # Forcibly delete CUDA tensors
        for obj in gc.get_objects():
            if isinstance(obj, pt.Tensor) and obj.is_cuda:
                obj.detach_().cpu()
                del obj

        pt.cuda.empty_cache()
        gc.collect()
