import copy
import pickle
import random

import numpy as np
import torch.nn as nn
import torch
from dataclasses import dataclass
from typing import Union, List, Tuple

from world.envs import OnePlayerEnv, TwoPlayerEnv, VersusBotEnv
from world.realm import Realm
from world.map_loaders.single_team import SingleTeamLabyrinthMapLoader, SingleTeamRocksMapLoader, SingleTeamMapLoader
from world.map_loaders.two_teams import TwoTeamRocksMapLoader, TwoTeamLabyrinthMapLoader
from world.scripted_agents import ClosestTargetAgent, Dummy
from world.utils import RenderedEnvWrapper
import tqdm


BATCH_SIZE = 32
GAMMA = 0.2
DEVICE = "cpu"
EPS_GREEDY = 0.07
EPOCHS = 60
LR = 1e-3
LA = 7e-1

PREHEAT_EPOCHS = 10
PREHEAT_DATASET_SIZE = 6000
DATASET_SIZE = 10000

# DQN
PER_EPOCH_TARGET_UPDATE = 5


@dataclass
class Data:
    state: np.array
    action: int
    next_state: np.array
    reward: int


class Agent(nn.Module):
    def __init__(self, action_dim: int = 5, state_dim: int = 40 * 40):
        super().__init__()
        self.Q_model = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=1, padding=0, stride=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(4, 6, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(6, 1, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(state_dim // 2**6, action_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.Q_model(x)

    def get_actions(self, state, info, history: np.array):
        res = []
        dummy, poss = dummy_move(state, info)
        for predator_i, predator_coord in enumerate(info["predators"]):
            state_i = convert_state(state, predator_coord, history[:, predator_i, :, :], flatten=False)
            action = self.Q_model(torch.from_numpy(state_i).float().to(DEVICE)).argmax().detach().cpu().numpy()
            if action in poss[predator_i]:
                res.append(action)
            else:
                res.append(action)
        return res

    def reset(self, state, info):
        pass

    #     self.Q_model = torch.load(__file__[:-8] + "/agent.pkl")


def prepare_history(state, info, history):
    steps = []
    for predator_i, predator_coord in enumerate(info["predators"]):
        if history is not None:
            step = convert_state(state, predator_coord, history[:, predator_i, :, :], False, unbiased=True)[2, :, :][np.newaxis, np.newaxis, :, :]
        else:
            step = convert_state(state, predator_coord, None, False, unbiased=True)[2, :, :][np.newaxis, np.newaxis, :, :]
        steps.append(step)
    res = np.concatenate(steps, axis=1)
    if history is not None:
        return np.concatenate([history, res], axis=0)[-5:, :, :, :]
    else:
        return res


def convert_state(state: np.array, predator_coord, history: np.array, flatten: bool = False, unbiased: bool = False):
    state_ = np.zeros((4, state.shape[0], state.shape[1]))
    num_teams = np.max(state[:, :, 0])

    state_[0, :, :][state[:, :, 1] == -1] = 1

    state_[1, :, :][state[:, :, 0] > 0] = 10
    state_[1, :, :][state[:, :, 0] == num_teams] = 3
    state_[2, :, :][state[:, :, 0] == 0] = 1
    if history is None:
        state_[3, :, :] = state_[2, :, :]
    else:
        state_[3, :, :] = np.sum(np.concatenate([history, state_[2, :, :][np.newaxis, :, :]]), axis=0)

    if not unbiased:
        state_ = np.roll(state_, state.shape[0] // 2 - predator_coord["y"], axis=1)
        state_ = np.roll(state_, state.shape[0] // 2 - predator_coord["x"], axis=2)
    if flatten:
        state_ = state_.flatten()

    return state_


def convert_response_to_data(
    state: np.array, prev_state: np.array, info, prev_info, action: np.array, history, flatten: bool = False, prev_victims=None
) -> Tuple[List[Data], int, np.array]:
    if prev_victims is None:
        prev_victims = np.array([0, 0, 0, 0, 0])
    res = []
    sum_ = []
    rewards = []

    num_teams = np.max(state[:, :, 0])
    for predator_i, predator_coord in enumerate(info["predators"]):
        # prepare state i
        state_i = convert_state(state, predator_coord, history[:, predator_i, :, :], flatten)

        # prepare prev state i
        prev_state_i = convert_state(prev_state, prev_info["predators"][predator_i], history[:, predator_i, :, :], flatten)

        # get action i
        action_i = action[predator_i]

        # new reward i - based on distance to victims
        reward_dist_i = np.zeros((state.shape[0], state.shape[1]))
        additions = np.array(list(range(20, 0, -1)) + list(range(1, 21)))[:, np.newaxis]
        reward_dist_i += additions
        reward_dist_i += additions.T
        reward_dist_i[state_i[1, :, :] == 0] = 0

        reward_dist_i_ = np.zeros((state.shape[0], state.shape[1]))
        additions = np.array(list(range(20, 0, -1)) + list(range(1, 21)))[:, np.newaxis]
        reward_dist_i_ += additions
        reward_dist_i_ += additions.T
        reward_dist_i_[prev_state_i[1, :, :] == 0] = 0

        reward_comp_i_dist = np.mean(reward_dist_i_[np.nonzero(reward_dist_i_)]) - np.mean(
            reward_dist_i[np.nonzero(reward_dist_i)]
        )

        # prepare reward i
        reward_comp_i = sum(
            [
                (1 if opp_team == num_teams else 3)
                for (opp_team, _), (team, x) in info["eaten"].items()
                if team == 0 and x == predator_i
            ]
        )
        _, poss = dummy_move(prev_state, prev_info)
        if action_i not in poss[predator_i]:
            reward_i = GAMMA * prev_victims[predator_i] - 5
        else:
            reward_i = GAMMA * prev_victims[predator_i] + (
                reward_comp_i * 10 if reward_comp_i > 0 else -2
            )

        sum_.append(reward_comp_i)
        rewards.append(reward_i)

        res.append(Data(state=prev_state_i, action=action_i, next_state=state_i, reward=reward_i))

    return res, sum(sum_), np.array(rewards)


def dummy_move(state, info):
    res = []
    check = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
    poss = []
    for predator_i, predator_coord in enumerate(info["predators"]):
        conv_state = convert_state(state, predator_coord, None, flatten=False)
        possibilities = []
        for i, ch in enumerate(check):
            if conv_state[0][20 + ch[0]][20 + ch[1]] != -1:
                possibilities.append(i)
        res.append(random.choice(possibilities))
        poss.append(possibilities)
    return np.array(res), poss


def sample_data(agent: Agent, env: Union[OnePlayerEnv, VersusBotEnv], rounds: int = 300, preheat: bool = False):
    state, info = env.reset()
    agent.reset(state, info)

    collected_data = []
    rewards = np.array([0, 0, 0, 0, 0])
    victims_ = 0

    to_unpack_backward = []

    history = prepare_history(state, info, None)

    for i in range(rounds):
        if random.random() < EPS_GREEDY:
            actions = np.random.randint(0, 5, 5)
        else:
            if preheat:
                actions = agent.get_actions(state, 0, history)
            else:
                actions = agent.get_actions(state, info, history)
        prev_state = state
        prev_info = info
        state, done, info = env.step(actions=actions)

        to_unpack_backward.append((state, prev_state, info, prev_info, actions, history))

        history = prepare_history(state, info, history)

        if done:
            break

    for state, prev_state, info, prev_info, actions, history in to_unpack_backward[::-1]:
        data, victims, rewards = convert_response_to_data(
            state, prev_state, info, prev_info, actions, history, prev_victims=rewards
        )
        victims_ += victims
        collected_data.extend(data)

    return collected_data, victims_


def build_dataset(
    agent: Agent, envs: List[Union[OnePlayerEnv, VersusBotEnv]], dataset_size: int, preheat: bool = False
):
    dataset = []
    information = []
    while len(dataset) < dataset_size:
        data, info = sample_data(agent, envs[random.randint(0, len(envs) - 1)], preheat=preheat)
        information.append(info)
        dataset.extend(data)
    return dataset[:dataset_size], np.mean(information)


def make_minibatches(dataset: List[Data], batch_size=64, shuffle=True):
    if shuffle:
        random.shuffle(dataset)
    for i in range(0, len(dataset), batch_size):
        slice_ = dataset[i : i + batch_size]
        states = np.stack([x.state for x in slice_], axis=0)
        actions = np.array([x.action for x in slice_])
        next_states = np.stack([x.next_state for x in slice_], axis=0)
        rewards = np.array([x.reward for x in slice_])

        yield {
            "state": torch.from_numpy(states).float().to(DEVICE),
            "action": torch.from_numpy(actions).long().to(DEVICE),
            "next_state": torch.from_numpy(next_states).float().to(DEVICE),
            "reward": torch.from_numpy(rewards).float().to(DEVICE),
        }


def single_train_epoch(
    agent: nn.Module, target_agent: nn.Module, opt: torch.optim.Optimizer, loss, dataset: List[Data]
):
    agent.train()
    losses = []
    for batch in make_minibatches(dataset, batch_size=BATCH_SIZE, shuffle=True):
        with torch.no_grad():
            next_action = agent(batch["next_state"]).argmax(1).unsqueeze(1)

        q_value = batch["reward"].unsqueeze(1) + GAMMA * target_agent(batch["next_state"]).gather(1, next_action)

        l = loss(agent(batch["state"]).gather(1, batch["action"].unsqueeze(1)), q_value)
        opt.zero_grad()
        l.backward()
        opt.step()
        losses.append(l.detach().cpu().numpy())
    return np.mean(losses)


def train(agent: nn.Module, envs: List[Union[OnePlayerEnv, VersusBotEnv, TwoPlayerEnv]]):
    target_agent = copy.deepcopy(agent)
    agent.to(DEVICE)
    target_agent.to(DEVICE)

    opt = torch.optim.Adam(agent.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=1e-4)

    # print("\nAgent preheat\n")
    # p_bar = tqdm.tqdm(range(PREHEAT_EPOCHS))
    # for i in p_bar:
    #     dataset, avg_eaten = build_dataset(ClosestTargetAgent(), envs, PREHEAT_DATASET_SIZE, preheat=True)
    #
    #     loss = single_train_epoch(agent, target_agent, opt, torch.nn.functional.mse_loss, dataset)
    #     p_bar.set_description(f"Avg loss: {loss}, Avg eaten: {avg_eaten}")
    #
    #     if i % PER_EPOCH_TARGET_UPDATE == 0:
    #         tau = 6e-1
    #         for target_param, local_param in zip(target_agent.parameters(), agent.parameters()):
    #             target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    #     torch.save(agent.Q_model, "agent.pkl")

    print("\nReal agent\n")
    p_bar = tqdm.tqdm(range(EPOCHS))
    for i in p_bar:
        agent.eval()
        dataset, avg_eaten = build_dataset(agent, envs, DATASET_SIZE)

        loss = single_train_epoch(agent, target_agent, opt, torch.nn.functional.mse_loss, dataset)
        p_bar.set_description(f"Avg loss: {loss}, Avg eaten: {avg_eaten}")
        scheduler.step()

        if i % PER_EPOCH_TARGET_UPDATE == 0:
            tau = 3e-1
            for target_param, local_param in zip(target_agent.parameters(), agent.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        torch.save(agent.Q_model, "agent.pkl")

    return agent


if __name__ == "__main__":
    # env1 = VersusBotEnv(Realm(Tw(), 1))
    env21 = VersusBotEnv(
        Realm(
            TwoTeamRocksMapLoader(rock_spawn_proba=0.01, additional_rock_spawn_proba=0.01),
            2,
            bots={1: ClosestTargetAgent()},
        )
    )
    env22 = VersusBotEnv(
        Realm(
            TwoTeamRocksMapLoader(rock_spawn_proba=0.1, additional_rock_spawn_proba=0.1),
            2,
            bots={1: ClosestTargetAgent()},
        )
    )
    env23 = VersusBotEnv(
        Realm(
            TwoTeamRocksMapLoader(rock_spawn_proba=0.15, additional_rock_spawn_proba=0.21),
            2,
            bots={1: ClosestTargetAgent()},
        )
    )
    env24 = VersusBotEnv(
        Realm(
            TwoTeamRocksMapLoader(rock_spawn_proba=0.15, additional_rock_spawn_proba=0.1),
            2,
            bots={1: ClosestTargetAgent()},
        )
    )
    env25 = VersusBotEnv(
        Realm(
            TwoTeamRocksMapLoader(rock_spawn_proba=0.01, additional_rock_spawn_proba=0.21),
            2,
            bots={1: ClosestTargetAgent()},
        )
    )
    env26 = VersusBotEnv(
        Realm(
            TwoTeamRocksMapLoader(rock_spawn_proba=0.1, additional_rock_spawn_proba=0.01),
            2,
            bots={1: ClosestTargetAgent()},
        )
    )
    env27 = VersusBotEnv(
        Realm(
            TwoTeamRocksMapLoader(rock_spawn_proba=0.15, additional_rock_spawn_proba=0.15),
            2,
            bots={1: ClosestTargetAgent()},
        )
    )
    env28 = VersusBotEnv(
        Realm(
            TwoTeamRocksMapLoader(rock_spawn_proba=0.10, additional_rock_spawn_proba=0.21),
            2,
            bots={1: ClosestTargetAgent()},
        )
    )
    env31 = VersusBotEnv(
        Realm(
            TwoTeamLabyrinthMapLoader(additional_links_min=1, additional_links_max=2),
            2,
            bots={1: ClosestTargetAgent()},
        )
    )
    env32 = VersusBotEnv(
        Realm(
            TwoTeamLabyrinthMapLoader(additional_links_min=2, additional_links_max=3),
            2,
            bots={1: ClosestTargetAgent()},
        )
    )
    env33 = VersusBotEnv(
        Realm(
            TwoTeamLabyrinthMapLoader(additional_links_min=4, additional_links_max=5),
            2,
            bots={1: ClosestTargetAgent()},
        )
    )
    env34 = VersusBotEnv(
        Realm(
            TwoTeamLabyrinthMapLoader(additional_links_min=5, additional_links_max=6),
            2,
            bots={1: ClosestTargetAgent()},
        )
    )
    env35 = VersusBotEnv(
        Realm(
            TwoTeamLabyrinthMapLoader(additional_links_min=7, additional_links_max=9),
            2,
            bots={1: ClosestTargetAgent()},
        )
    )
    env36 = VersusBotEnv(
        Realm(
            TwoTeamLabyrinthMapLoader(additional_links_min=10, additional_links_max=11),
            2,
            bots={1: ClosestTargetAgent()},
        )
    )
    env37 = VersusBotEnv(
        Realm(
            TwoTeamLabyrinthMapLoader(additional_links_min=11, additional_links_max=12),
            2,
            bots={1: ClosestTargetAgent()},
        )
    )
    env38 = VersusBotEnv(
        Realm(
            TwoTeamLabyrinthMapLoader(additional_links_min=12, additional_links_max=12),
            2,
            bots={1: ClosestTargetAgent()},
        )
    )

    agent = Agent()

    agent = train(
        agent,
        [
            env21,
            env22,
            env23,
            env24,
            env25,
            env26,
            env27,
            env28,
            env31,
            env32,
            env33,
            env34,
            env35,
            env36,
            env37,
            env38,
        ],
    )
    # agent = train(
    #     agent,
    #     [
    #         env26,
    #         env27,
    #         env28,
    #         env36,
    #         env37,
    #         env38,
    #     ],
    # )

    torch.save(agent.Q_model, "agent.pkl")
