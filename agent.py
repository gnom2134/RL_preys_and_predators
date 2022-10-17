# class Agent:
#     def get_actions(self, state, info):
#         return [0, 0, 0, 0, 0]
#
#     def reset(self, state, info):
#         pass


import torch.nn as nn
import torch
import numpy as np
import random


def convert_state(state: np.array, predator_coord, flatten: bool = False):
    state_ = np.zeros((3, state.shape[0], state.shape[1]))
    num_teams = np.max(state[:, :, 0])

    state_[0, :, :][state[:, :, 1] == -1] = 1

    state_[1, :, :][state[:, :, 0] > 0] = 10
    state_[1, :, :][state[:, :, 0] == num_teams] = 3
    state_[2, :, :][state[:, :, 0] == 0] = 1

    state_ = np.roll(state_, state.shape[0] // 2 - predator_coord["y"], axis=1)
    state_ = np.roll(state_, state.shape[0] // 2 - predator_coord["x"], axis=2)
    if flatten:
        state_ = state_.flatten()

    return state_


def dummy_move(state, info):
    res = []
    check = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
    poss = []
    for predator_i, predator_coord in enumerate(info["predators"]):
        conv_state = convert_state(state, predator_coord, flatten=False)
        possibilities = []
        for i, ch in enumerate(check):
            if conv_state[0][20 + ch[0]][20 + ch[1]] != -1:
                possibilities.append(i)
        res.append(random.choice(possibilities))
        poss.append(possibilities)
    return np.array(res), poss


class Agent(nn.Module):
    def __init__(self, action_dim: int = 5, state_dim: int = 40 * 40):
        super().__init__()
        self.Q_model = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=5, padding=2, stride=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(state_dim // 2**4, action_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.Q_model(x)

    def get_actions(self, state, info):
        res = []
        dummy, poss = dummy_move(state, info)
        for predator_i, predator_coord in enumerate(info["predators"]):
            state_i = convert_state(state, predator_coord, flatten=False)
            action = self.Q_model(torch.from_numpy(state_i).float()).argmax().detach().cpu().numpy()
            if action in poss[predator_i]:
                res.append(action)
            else:
                res.append(action)
        return res

    def reset(self, state, info):
        self.Q_model = torch.load(__file__[:-8] + "/agent.pkl")
