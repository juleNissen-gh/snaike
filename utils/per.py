"""
Enhanced Prioritized Experience Replay memory with save/load capabilities
"""
from collections import namedtuple
import torch as pt
import numpy as np
import os

# Replay memory
Staticmem = namedtuple('Staticmem',
                       ('states', 'b_matricies', 'next_states', 'next_b_matricies', 'rewards', 'playing', 'weights'))

class PrioritizedReplayMemory:
    """
    A class implementing Prioritized Experience Replay (PER) memory.

    This memory stores experiences and samples them based on their priority,
    allowing more important experiences to be sampled more frequently.
    """

    def __init__(self,
                 mem_len: int,
                 input_len: int,
                 truncated_matrix_size: int,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.0003,
                 beta_end: float = 1,
                 device: pt.device = pt.device('cpu')):
        """
        Initialize the PrioritizedReplayMemory.

        Args:
            mem_len (int): The maximum number of experiences to store.
            input_len (int): The length of the input state vector.
            truncated_matrix_size (int): The size of the truncated board matrix.
            alpha (float): The exponent determining how much prioritization is used. Defaults to 0.6.
            beta (float): The initial value of beta for importance sampling. Defaults to 0.4.
            beta_increment (float): The increment of beta per sampling. Defaults to 0.0003.
            beta_end (float): The final value of beta. Defaults to 1.
            device (torch.device): The device to store the tensors on. Defaults to CPU.
        """
        self.device = device
        self.capacity = mem_len
        self.num_blanc_values = mem_len
        self.input_len = input_len
        self.truncated_matrix_size = truncated_matrix_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.beta_end = beta_end

        # Initialize memory tensors
        self.states = pt.zeros((self.capacity, self.input_len), dtype=pt.int8)
        self.b_matricies = pt.zeros((self.capacity, self.truncated_matrix_size, self.truncated_matrix_size),
                                    dtype=pt.bool).to_sparse_coo()
        self.next_states = pt.zeros((self.capacity, 4, self.input_len), dtype=pt.int8)
        self.next_b_matricies = pt.zeros((self.capacity, 4, self.truncated_matrix_size, self.truncated_matrix_size),
                                         dtype=pt.bool).to_sparse_coo()
        self.rewards = pt.zeros((self.capacity, 4), dtype=pt.half)
        self.playing = pt.zeros((self.capacity, 4), dtype=pt.bool)
        self.weights = pt.ones(self.capacity, dtype=pt.float32)
        self.priorities = pt.ones(self.capacity, dtype=pt.float32)

    def save(self, filepath: str):
        """
        Save the replay memory to disk.

        Args:
            filepath (str): Path where to save the memory
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Move everything to CPU for saving
        save_dict = {
            'states': self.states.cpu(),
            'b_matricies': self.b_matricies.cpu(),
            'next_states': self.next_states.cpu(),
            'next_b_matricies': self.next_b_matricies.cpu(),
            'rewards': self.rewards.cpu(),
            'playing': self.playing.cpu(),
            'weights': self.weights.cpu(),
            'priorities': self.priorities.cpu(),
            'num_blanc_values': self.num_blanc_values,
            # Save parameters
            'params': {
                'capacity': self.capacity,
                'input_len': self.input_len,
                'truncated_matrix_size': self.truncated_matrix_size,
                'alpha': self.alpha,
                'beta': self.beta,
                'beta_increment': self.beta_increment,
                'beta_end': self.beta_end
            }
        }
        pt.save(save_dict, filepath)

    @classmethod
    def load(cls, filepath: str, device: pt.device = None):
        """
        Load a replay memory from disk.

        Args:
            filepath (str): Path to the saved memory file
            device (torch.device): Device to load the memory to

        Returns:
            PrioritizedReplayMemory: Loaded memory object
        """
        save_dict = pt.load(filepath)
        params = save_dict['params']

        # Create new instance
        memory = cls(
            mem_len=params['capacity'],
            input_len=params['input_len'],
            truncated_matrix_size=params['truncated_matrix_size'],
            alpha=params['alpha'],
            beta=params['beta'],
            beta_increment=params['beta_increment'],
            beta_end=params['beta_end'],
            device=device if device is not None else pt.device('cpu')
        )

        # Load states
        memory.states = save_dict['states'].to(device=memory.device)
        memory.b_matricies = save_dict['b_matricies'].to(device=memory.device)
        memory.next_states = save_dict['next_states'].to(device=memory.device)
        memory.next_b_matricies = save_dict['next_b_matricies'].to(device=memory.device)
        memory.rewards = save_dict['rewards'].to(device=memory.device)
        memory.playing = save_dict['playing'].to(device=memory.device)
        memory.weights = save_dict['weights'].to(device=memory.device)
        memory.priorities = save_dict['priorities'].to(device=memory.device)
        memory.num_blanc_values = save_dict['num_blanc_values']

        print(f"Loaded replay memory from {filepath}")
        print(f"Memory contains {memory.capacity - memory.num_blanc_values} experiences")
        return memory

    def get_stats(self) -> dict:
        """
        Get statistics about the stored experiences.

        Returns:
            dict: Statistics about the memory contents
        """
        valid_experiences = self.capacity - self.num_blanc_values
        return {
            'total_experiences': valid_experiences,
            'reward_mean': self.rewards[:valid_experiences].mean().item(),
            'reward_std': self.rewards[:valid_experiences].std().item(),
            'positive_rewards': (self.rewards[:valid_experiences] > 0).sum().item(),
            'negative_rewards': (self.rewards[:valid_experiences] < 0).sum().item(),
            'terminal_states': (~self.playing[:valid_experiences]).sum().item()
        }

    # Original methods remain the same
    def push(self, batch: Staticmem):
        """Add a batch of experiences to the memory."""
        length = batch.states.shape[0]
        self.num_blanc_values = max(0, self.num_blanc_values - length)

        # Update attributes
        update_slicer = slice(None, self.capacity - length)

        self.states = pt.cat((batch.states, self.states[update_slicer]))
        self.next_states = pt.cat((batch.next_states, self.next_states[update_slicer]))

        new_b_matrices = pt.zeros_like(self.b_matricies.to_dense())
        new_b_matrices[:length] = batch.b_matricies
        new_b_matrices[length:] = self.b_matricies.to_dense()[update_slicer]
        self.b_matricies = new_b_matrices.to_sparse_coo()

        new_next_b_matrices = pt.zeros_like(self.next_b_matricies.to_dense())
        new_next_b_matrices[:length] = batch.next_b_matricies
        new_next_b_matrices[length:] = self.next_b_matricies.to_dense()[update_slicer]
        self.next_b_matricies = new_next_b_matrices.to_sparse_coo()

        self.rewards = pt.cat((batch.rewards, self.rewards[update_slicer]))
        self.playing = pt.cat((batch.playing, self.playing[update_slicer]))
        self.weights = pt.cat((batch.weights, self.weights[update_slicer]))

        max_prio = max(self.priorities.max().item(), 1)
        self.priorities = pt.cat(
            (pt.full((length,), max_prio, dtype=pt.float32), self.priorities[update_slicer]))

        return self

    def sample(self, batch_size: int) -> tuple[Staticmem, np.ndarray]:
        """Sample a batch of experiences based on their priorities."""
        probabilities: np.ndarray = \
            (self.priorities.cpu().numpy() ** self.alpha)[:(vals := self.capacity - self.num_blanc_values)]

        probabilities /= probabilities.sum()
        indices = np.random.choice(vals, min(batch_size, vals), p=probabilities)

        self.beta = min(self.beta_end, self.beta + self.beta_increment)
        weights = (self.capacity * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = pt.from_numpy(weights.astype(np.float32)).to(device=self.device)

        batch = Staticmem(
            self.states[indices].float(),
            self.b_matricies.to_dense()[indices].float(),
            self.next_states[indices].float(),
            self.next_b_matricies.to_dense()[indices].float(),
            self.rewards[indices].float(),
            self.playing[indices],
            weights
        )

        return batch, indices

    def update_priorities(self, indices: np.ndarray, td_errors: pt.Tensor) -> None:
        """Update priorities for sampled experiences."""
        priorities = (td_errors.abs() + 1e-5).pow(self.alpha)
        self.priorities[indices] = priorities
        self.weights[indices] = priorities.pow(-self.beta)
        self.weights /= self.weights.max()
        del td_errors, priorities

    def __iter__(self) -> pt.Tensor:
        """
        Iterate over the memory attributes.

        Yields:
            torch.Tensor: The memory attributes in the order: states, b_matricies, next_states,
                          next_b_matricies, rewards, playing, weights.
        """
        yield self.states.float()
        yield self.b_matricies
        yield self.next_states.float()
        yield self.next_b_matricies
        yield self.rewards
        yield self.playing
        yield self.weights
