from collections import deque

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from shared_adam import SharedAdam


class Actor(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)

        return x


class Critic(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    # 当前时刻的state_value
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self):
        transitions = self.memory
        states, action_probs, next_states, rewards, dones = zip(*transitions)

        states = np.vstack(states)
        action_probs = np.vstack(action_probs)
        next_states = np.vstack(next_states)
        rewards = np.vstack(rewards)
        dones = np.vstack(dones)

        states = torch.tensor(states, dtype=torch.float32)
        action_probs = torch.tensor(action_probs, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        return states, action_probs, next_states, rewards, dones

    def clr(self):
        self.memory.clear()


def update_model(source, target, tau):
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


class MAA2C(nn.Module):
    def __init__(self, n_agents, obs_dim, action_dim, num_s, style=True):
        super().__init__()

        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.lr = 1e-5
        self.gamma = 0.99
        self.tau = 1e-5
        self.eps = 0.2

        self.epochs = num_s

        # initial memory
        self.memory = ReplayMemory(capacity=5000)
        self.batch_size = None

        # initial actors
        self.actors = nn.ModuleList([Actor(self.obs_dim, self.action_dim) for _ in range(self.n_agents)])
        self.target_actors = nn.ModuleList([Actor(self.obs_dim, self.action_dim) for _ in range(self.n_agents)])
        for i in range(self.n_agents):
            update_model(self.actors[i], self.target_actors[i], tau=1.0)

        # initial critics
        self.critics = nn.ModuleList(
            [Critic(self.obs_dim * self.n_agents, self.action_dim) for _ in range(self.n_agents)])
        self.target_critics = nn.ModuleList(
            [Critic(self.obs_dim * self.n_agents, self.action_dim) for _ in range(self.n_agents)])
        for i in range(self.n_agents):
            update_model(self.critics[i], self.target_critics[i], tau=1.0)

        self.style = style
        if self.style:
            for i in range(self.n_agents):
                self.actors[i].share_memory()
                self.critics[i].share_memory()
            self.actors_optimizer = [SharedAdam(self.actors[i].parameters(), lr=self.lr) for i in range(self.n_agents)]
            self.critics_optimizer = [SharedAdam(self.critics[i].parameters(), lr=self.lr) for i in
                                      range(self.n_agents)]

        else:
            self.actors_optimizer = [torch.optim.Adam(self.actors[i].parameters(), lr=self.lr) for i in
                                     range(self.n_agents)]
            self.critics_optimizer = [torch.optim.Adam(self.critics[i].parameters(), lr=self.lr) for i in
                                      range(self.n_agents)]

        self.mse_loss = nn.MSELoss()

    def select_action(self, obs, i):
        obs = torch.from_numpy(obs).float()

        probs = self.actors[i](obs)
        action_dist = torch.distributions.Categorical(probs)
        # 从当前概率分布中随机采样tensor-->int
        action = action_dist.sample().item()
        return action

    def push(self, transition):
        before_state, action_probs, state, rewards, done = transition
        before_state = before_state.flatten()
        action_probs = np.array(action_probs).flatten()
        state = state.flatten()
        # 按行压平
        transition = (before_state, action_probs, state, rewards, done)
        self.memory.push(transition)

    def train_model(self, actor_optimizers, critic_optimizers, actors, critics):  # only update agent i's network

        self.batch_size = len(self.memory)

        states, actions, next_states, rewards, dones = self.memory.sample()

        # make target q values
        for i in range(self.n_agents):

            Q_target = self.target_critics[i](states).detach()

            r = torch.zeros(self.memory.__len__())

            for t in range(len(rewards[:, i])):  # reverse buffer r

                if dones.squeeze()[t, i]:
                    r[t] = rewards[:, i][t]
                else:
                    r[t] = rewards[:, i][t] + self.gamma * Q_target[t + 1]

            current_q = self.critics[i](states).squeeze().detach()

            advantage = r - current_q

            probs = self.actors[i](states.view(self.batch_size, -1, self.obs_dim)[:, i])
            old_log_probs = torch.log(probs.gather(1, actions[:, i].type(torch.long).unsqueeze(1))).detach()
            # 一组数据训练 epochs 轮
            for _ in range(self.epochs):
                # 每一轮更新一次策略网络预测的状态
                probs = self.actors[i](states.view(self.batch_size, -1, self.obs_dim)[:, i])
                log_probs = torch.log(probs.gather(1, actions[:, i].type(torch.long).unsqueeze(1)))
                # 新旧策略之间的比例
                ratio = torch.exp(log_probs - old_log_probs)
                # 近端策略优化裁剪目标函数公式的左侧项
                surr1 = ratio * advantage
                # 公式的右侧项，ratio小于(1-eps)就输出(1-eps)，大于(1+eps)就输出(1+eps)
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

                # update actor network
                actor_loss = torch.mean(-torch.min(surr1, surr2))

                self.actors_optimizer[i].zero_grad()
                actor_optimizers[i].zero_grad()
                actor_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 5)
                torch.nn.utils.clip_grad_norm_(actors[i].parameters(), 5)
                for lp, gp in zip(self.actors[i].parameters(), actors[i].parameters()):
                    gp._grad = lp.grad
                self.actors_optimizer[i].step()
                actor_optimizers[i].step()

                # update critic network with MSE loss
                current_q = self.critics[i](states).squeeze()
                value_loss = self.mse_loss(r, current_q)

                self.critics_optimizer[i].zero_grad()
                critic_optimizers[i].zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 5)
                torch.nn.utils.clip_grad_norm_(critics[i].parameters(), 5)
                for lp, gp in zip(self.critics[i].parameters(), critics[i].parameters()):
                    gp._grad = lp.grad
                self.critics_optimizer[i].step()
                critic_optimizers[i].step()

        # pull global parameters and update target network
        for i in range(self.n_agents):
            # pull global parameters
            self.actors[i].load_state_dict(actors[i].state_dict())
            self.critics[i].load_state_dict(critics[i].state_dict())

            # update target network
            update_model(self.actors[i], self.target_actors[i], self.tau)
            update_model(self.critics[i], self.target_critics[i], self.tau)

    def save_model(self, file_name):
        torch.save(self.state_dict(), file_name)

    def load_model(self, file_name):
        self.load_state_dict(torch.load(file_name))
