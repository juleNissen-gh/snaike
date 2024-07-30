"""
Prioritized experience replay memory class and Staticmem namedtuple
"""
from collections import namedtuple
import torch as pt
import numpy as np

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

        self.states = pt.zeros((mem_len, input_len), dtype=pt.int8)
        self.b_matricies = pt.zeros((mem_len, truncated_matrix_size, truncated_matrix_size),
                                    dtype=pt.bool).to_sparse_coo()
        self.next_states = pt.zeros((mem_len, 4, input_len), dtype=pt.int8)
        self.next_b_matricies = pt.zeros((mem_len, 4, truncated_matrix_size, truncated_matrix_size),
                                         dtype=pt.bool).to_sparse_coo()
        self.rewards = pt.zeros((mem_len, 4), dtype=pt.half)
        self.playing = pt.zeros((mem_len, 4), dtype=pt.bool)
        self.weights = pt.ones(mem_len, dtype=pt.float32)
        self.priorities = pt.ones(mem_len, dtype=pt.float32)

    def push(self, batch: Staticmem):
        """
        Add a batch of experiences to the memory.

        Args:
            batch (Staticmem): A named tuple containing a batch of experiences.

        Returns:
            PrioritizedReplayMemory: The updated memory object.
        """
        length = batch.states.shape[0]
        self.num_blanc_values = max(0, self.num_blanc_values - length)

        # Update attributes
        update_slicer = slice(None, self.capacity - length)

        self.states = pt.cat((batch.states, self.states[update_slicer]))
        self.next_states = pt.cat((batch.next_states, self.next_states[update_slicer]))

        self.b_matricies = self.b_matricies.to_dense()
        self.b_matricies = pt.cat((batch.b_matricies, self.b_matricies[update_slicer]))
        self.b_matricies = self.b_matricies.to_sparse_coo()

        self.next_b_matricies = self.next_b_matricies.to_dense()
        self.next_b_matricies = pt.cat((batch.next_b_matricies, self.next_b_matricies[update_slicer]))
        self.next_b_matricies = self.next_b_matricies.to_sparse_coo()

        self.rewards = pt.cat((batch.rewards, self.rewards[update_slicer]))
        self.playing = pt.cat((batch.playing, self.playing[update_slicer]))
        self.weights = pt.cat((batch.weights, self.weights[update_slicer]))

        max_prio = max(self.priorities.max().item(), 1)
        self.priorities = pt.cat(
            (pt.full((length,), max_prio, dtype=pt.float32), self.priorities[update_slicer]))

        assert length < self.capacity, f'{length=}\n{max_prio=}'

        return self

    def sample(self, batch_size: int) -> tuple[Staticmem, np.ndarray]:
        """
        Sample a batch of experiences from the memory.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            - batch (Staticmem): A named tuple of sampled experiences.
            - indices (numpy.ndarray): The indices of the sampled experiences.
        """
        # get the probabilies for sampling
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
        """
        Update the priorities of the sampled experiences.

        Args:
            indices (numpy.ndarray): The indices of the experiences to update.
            td_errors (torch.Tensor): The TD errors of the sampled experiences.
        """
        priorities = (td_errors.abs() + 1e-5).pow(self.alpha)
        self.priorities[indices] = priorities
        self.weights[indices] = priorities.pow(-self.beta)
        self.weights /= self.weights.max()

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
