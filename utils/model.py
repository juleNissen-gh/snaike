import numpy as np
import random
import torch as pt
from torch import nn
from utils.per import Staticmem


class Model(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 matrix_size: int,
                 matrix2_size: int,
                 ) -> None:
        super().__init__()

        # network params
        self.output_size: int = output_size
        self.matrix_size: int = matrix_size
        self.matrix2_size: int = matrix2_size
        self.conv_ochannels: int = 8
        self.conv2_ochannels: int = 4

        self.conv: nn.Module = nn.Sequential(  # conv layers
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, self.conv_ochannels, 3, padding=1),
            nn.ReLU()
        )

        self.conv2: nn.Module = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, self.conv2_ochannels, 3, padding=1),
            nn.ReLU(),
        )

        # calculate output size of convnet
        self.cnn_output_size = self.conv(pt.zeros(1, self.matrix_size, self.matrix_size)).shape[-1]
        self.cnn2_output_size = self.conv2(pt.zeros(1, self.matrix2_size, self.matrix2_size)).shape[-1]

        self.leaky_relu2 = nn.LeakyReLU()
        self.leaky_relu4 = nn.LeakyReLU()
        self.leaky_relu6 = nn.LeakyReLU()
        self.leaky_relu8 = nn.LeakyReLU()
        self.leaky_relu10 = nn.LeakyReLU()

        self.linear1 = nn.Linear(input_size, prev_o := hidden_size)
        self.linear3 = nn.Linear(prev_o + self.cnn2_output_size ** 2 * self.conv2_ochannels, prev_o := hidden_size * 2)
        self.linear5 = nn.Linear(prev_o + self.cnn_output_size ** 2 * self.conv_ochannels, prev_o := hidden_size * 3)
        self.linear7 = nn.Linear(prev_o, prev_o := hidden_size * 2)
        self.linear9 = nn.Linear(prev_o + hidden_size, prev_o := hidden_size)
        self.linear11 = nn.Linear(prev_o, output_size)

    def forward(self, x: pt.Tensor, board_matrix: pt.Tensor) -> pt.Tensor:
        out_conv: pt.Tensor = self.conv(board_matrix.float().unsqueeze(-3))

        pad = (self.matrix_size - self.matrix2_size) // 2
        out_conv2: pt.Tensor = self.conv2(board_matrix[..., pad:-pad, pad:-pad].float().unsqueeze(-3))

        res_conn = self.linear1(x)
        x = self.leaky_relu2(res_conn)
        x = self.linear3(pt.cat((x, pt.flatten(out_conv2, -3)), -1))
        x = self.leaky_relu4(x)
        x = self.linear5(pt.cat((x, pt.flatten(out_conv, -3)), -1))
        x = self.leaky_relu6(x)
        x = self.linear7(x)
        x = self.leaky_relu8(x)
        x = self.linear9(pt.cat((x, res_conn), -1))
        x = self.leaky_relu10(x)
        x = self.linear11(x)

        return x

    def select_action(self, state: tuple[pt.Tensor, pt.Tensor], epsilon: float) -> pt.Tensor:
        if random.random() > epsilon:
            with pt.no_grad():
                return pt.tensor([[pt.argmax(self(*state)).item()]])
        else:
            return pt.tensor([[random.randint(0, self.output_size - 1)]])

    def to_matrix(self, board_pos: np.ndarray) -> pt.Tensor:
        board_matrix = pt.zeros((self.board_size,) * 2).to(dtype=pt.bool)
        board_matrix[board_pos[:, 0], board_pos[:, 1]] = 1  # populating the matrix
        return board_matrix


def compute_q_values(batch: Staticmem, policy_net, target_net, gamma):
    # These are the values which would've been calculated for each batch state
    state_action_values = policy_net(batch.states, batch.b_matricies)

    next_state_values = pt.zeros(batch.playing.shape)
    # Compute V(s_{t+1}) for all next non-final states.
    with pt.no_grad():
        next_state_values[batch.playing] = \
            pt.max(target_net(batch.next_states[batch.playing], batch.next_b_matricies[batch.playing]), 1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + batch.rewards

    return state_action_values, expected_state_action_values


def optimize(*, policy_net, target_net, memory, criterion, optimizer, batch_size, gamma, tau):
    batch, indices = memory.sample(batch_size)
    computed_qvalues, expected_qvalues = compute_q_values(batch, policy_net, target_net, gamma)

    loss = criterion(computed_qvalues, expected_qvalues)
    weighted_loss = (loss * batch.weights.unsqueeze(-1)).mean()

    weighted_loss.backward()
    pt.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    optimizer.zero_grad()

    td_errors = (computed_qvalues - expected_qvalues).abs().mean(1).detach()
    memory.update_priorities(indices, td_errors)

    # Soft update target net
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

    return weighted_loss.item()
