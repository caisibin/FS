import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from shared_adam import SharedAdam


class Memory:
    def __init__(self, agent_num, action_dim):
        self.agent_num = agent_num
        self.action_dim = action_dim

        self.actions = []
        self.observations = []
        self.pi = [[] for _ in range(agent_num)]
        self.reward = []
        self.done = [[] for _ in range(agent_num)]

    def get(self):
        actions = torch.tensor(self.actions)
        observations = torch.cat(self.observations).reshape(-1, 8, 10)

        pi = []
        for i in range(self.agent_num):
            pi.append(torch.cat(self.pi[i]).view(len(self.pi[i]), self.action_dim))

        reward = torch.tensor(np.array(self.reward))
        done = self.done

        return actions, observations, pi, reward, done

    def clear(self):
        self.actions = []
        self.observations = []
        self.pi = [[] for _ in range(self.agent_num)]
        self.reward = []
        self.done = [[] for _ in range(self.agent_num)]


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return F.softmax(self.fc3(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, agent_num, obs_dim):
        super().__init__()
        self.agent_num = agent_num
        self.obs_dim = obs_dim

        input_dim = self.obs_dim * self.agent_num + self.agent_num

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, observations, actions):
        observations = observations.view(-1, self.obs_dim * self.agent_num)

        x = torch.cat([observations.type(torch.float32), actions.type(torch.float32)], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class COMA(nn.Module):
    def __init__(self, agent_num, state_dim, action_dim, lr_c, lr_a, gamma, target_update_steps, num_s, style=True):
        super().__init__()
        self.agent_num = agent_num
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_c = lr_c
        self.lr_a = lr_a
        self.gamma = gamma
        self.target_update_steps = target_update_steps

        self.memory = Memory(agent_num, action_dim)

        self.actors = nn.ModuleList([Actor(state_dim, action_dim) for _ in range(agent_num)])
        self.critics = nn.ModuleList([Critic(agent_num, state_dim) for _ in range(agent_num)])

        self.style = style
        if self.style:
            for i in range(agent_num):
                self.actors[i].share_memory()
                self.critics[i].share_memory()
            self.actors_optimizer = [SharedAdam(self.actors[i].parameters(), lr=self.lr_a) for i in range(agent_num)]
            self.critics_optimizer = [SharedAdam(self.critics[i].parameters(), lr=self.lr_c) for i in range(agent_num)]

        else:
            self.actors_optimizer = [torch.optim.Adam(self.actors[i].parameters(), lr=self.lr_a) for i in
                                     range(agent_num)]
            self.critics_optimizer = [torch.optim.Adam(self.critics[i].parameters(), lr=self.lr_c) for i in
                                      range(agent_num)]

        self.critics_target = nn.ModuleList([Critic(agent_num, state_dim) for _ in range(agent_num)])
        for i in range(agent_num):
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())

        self.count = 0
        self.epochs = num_s
        self.eps = 0.2

    def get_actions(self, observations, train=True):

        observations = torch.from_numpy(observations).float()

        actions = []

        for i in range(self.agent_num):

            dist = self.actors[i](observations[i])
            self.memory.pi[i].append(dist)

            action = Categorical(dist).sample()
            if not train:
                action = torch.argmax(dist)

            actions.append(action.item())

        self.memory.observations.append(observations)
        self.memory.actions.append(actions)

        return actions

    def train_model(self, actors_optimizer, critics_optimizer, actors, critics):

        actions, observations, pi, reward, done = self.memory.get()

        for i in range(self.agent_num):
            # train actor
            Q_primes = []

            Q = self.critics[i](observations, actions).squeeze().detach()
            actions_prime = actions.clone()
            for j in range(self.action_dim):
                actions_prime[:, i] = j

                Q_prime = self.critics[i](observations, actions_prime)

                Q_primes.append(Q_prime)
            Q_primes = torch.cat(Q_primes, dim=1)
            baseline = torch.sum(pi[i] * Q_primes, dim=1).detach()
            advantage = Q - baseline
            action_taken = actions.type(torch.long)[:, i].reshape(-1, 1)
            old_log_probs = torch.log(torch.gather(pi[i], dim=1, index=action_taken).squeeze()).detach()

            Q_target = self.critics_target[i](observations, actions).detach()

            # TD(0)
            r = torch.zeros(len(reward[:, i]))
            for t in range(len(reward[:, i])):

                if done[i][t]:
                    r[t] = reward[:, i][t]
                else:
                    r[t] = reward[:, i][t] + self.gamma * Q_target[t + 1]

            # 一组数据训练 epochs 轮
            for _ in range(self.epochs):
                # 每一轮更新一次策略网络预测的状态

                log_probs = torch.log(
                    torch.gather(self.actors[i](observations[:, i]), dim=1, index=action_taken).squeeze())
                # 新旧策略之间的比例
                ratio = torch.exp(log_probs - old_log_probs)
                # 近端策略优化裁剪目标函数公式的左侧项
                surr1 = ratio * advantage
                # 公式的右侧项，ratio小于(1-eps)就输出(1-eps)，大于(1+eps)就输出(1+eps)
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

                # 策略网络的损失函数
                actor_loss = torch.mean(-torch.min(surr1, surr2))

                # 价值网络的损失函数
                Q = self.critics[i](observations, actions).squeeze()
                critic_loss = torch.mean((r - Q) ** 2)

                # calculate local gradients and push local parameters to global
                # train actor
                self.actors_optimizer[i].zero_grad()
                actors_optimizer[i].zero_grad()

                actor_loss.backward()
                for lp, gp in zip(self.actors[i].parameters(), actors[i].parameters()):
                    gp._grad = lp.grad

                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 5)
                torch.nn.utils.clip_grad_norm_(actors[i].parameters(), 5)

                self.actors_optimizer[i].step()
                actors_optimizer[i].step()

                # train critic
                self.critics_optimizer[i].zero_grad()
                critics_optimizer[i].zero_grad()

                critic_loss.backward()
                for lp, gp in zip(self.critics[i].parameters(), critics[i].parameters()):
                    gp._grad = lp.grad

                torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 5)
                torch.nn.utils.clip_grad_norm_(critics[i].parameters(), 5)

                self.critics_optimizer[i].step()
                critics_optimizer[i].step()

        # pull global parameters
        for i in range(self.agent_num):
            self.actors[i].load_state_dict(actors[i].state_dict())
            self.critics[i].load_state_dict(critics[i].state_dict())

        if self.count % self.target_update_steps == 0:
            for i in range(self.agent_num):
                self.critics_target[i].load_state_dict(self.critics[i].state_dict())

        self.count += 1
        self.memory.clear()

    def save_model(self, file_name):

        torch.save(self.state_dict(), file_name)

    def load_model(self, file_name):
        self.load_state_dict(torch.load(file_name))
