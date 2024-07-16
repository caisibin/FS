import argparse
import multiprocessing as mp
import os
import sys
import time

import pandas as pd

from COMA import COMA
from env import TrafficEnv
from utils import record

parser = argparse.ArgumentParser()
parser.add_argument("-R", "--render", action="store_true",
                    help="whether render while training or not")
parser.add_argument("--obs_dim", help="")
parser.add_argument("--action_dim", help="")
parser.add_argument("--n_phase", help="")
parser.add_argument("--n_agents", help="")
parser.add_argument("--n_intersections", help="")
parser.add_argument("--n_episode", help="")
parser.add_argument("--decision_time", help="")
args = parser.parse_args()

# Hyperparameters
lr_c = 0.005
lr_a = 0.0001
gamma = 0.99


class Worker(mp.Process):
    def __init__(self, agents, episode_reward, episodes_reward, episode, order, average_traveling_time,
                 performance_list, e, num_p, num_s, k, min_time):
        super().__init__()
        self.performance_list = performance_list
        self.average_traveling_time = average_traveling_time
        self.agents = agents
        self.episode_reward = episode_reward
        self.episodes_reward = episodes_reward
        self.episode = episode
        self.name = 'w%02i' % order
        self.agents_local = COMA(agent_num, obs_dim, action_dim, lr_c, lr_a, gamma, target_update_steps, num_s, False)
        # Create an Environment and RL Agent
        self.env = TrafficEnv(args, self.name, mode="gui") if args.render else TrafficEnv(args, self.name)
        self.e = e

        self.num_p = num_p
        self.num_s = num_s
        self.k = k

        self.min_time = min_time

    def run(self):

        ep_r = 0

        obs = self.env.reset()

        while self.episode.value <= 330 * self.num_p:

            actions = self.agents_local.get_actions(obs)

            next_obs, reward, done_n = self.env.step(actions)

            self.agents_local.memory.reward.append(reward)

            for i in range(agent_num):
                self.agents_local.memory.done[i].append(done_n[i])

            ep_r += sum(reward)

            obs = next_obs

            if all(done_n):
                self.env.close()

                self.agents_local.train_model(self.agents.actors_optimizer, self.agents.critics_optimizer,
                                              self.agents.actors, self.agents.critics)

                record(self.episode, self.episode_reward, ep_r, self.episodes_reward,
                       self.average_traveling_time, self.performance_list, self.name, self.e)
                if self.name == 'w07':
                    with self.min_time.get_lock():
                        if self.average_traveling_time.value < self.min_time.value:
                            self.min_time.value = self.average_traveling_time.value
                            # Save the model
                            self.agents_local.save_model(f"results/th/{self.num_s}_{self.num_p}_{self.k}.th")
                ep_r = 0

                obs = self.env.reset()

        self.env.close()

        self.episodes_reward.put(None)
        self.performance_list.put(None)


if __name__ == '__main__':
    print('---training start---')
    start = time.time()
    count = 0
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    # for num_s, num_p in [[1, 1], [5, 1], [10, 1], [15, 1], [20, 1], [5, 2], [1, 3], [1, 2], [20, 2]]:

    # for num_s, num_p in [[1, 1], [5, 1], [10, 1], [15, 1], [20, 1]]:
    for num_s, num_p in [[5, 2], [1, 3], [1, 2], [20, 2]]:
        min_time = mp.Value('d', 10000)
        for k in range(2):
            print('---traning---')
            print(f"num_s={num_s}, num_p={num_p}, {k}th")
            count += 1

            # Before the start, should check SUMO_HOME is in your environment variables
            if 'SUMO_HOME' in os.environ:
                tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
                sys.path.append(tools)
            else:
                sys.exit("please declare environment variable 'SUMO_HOME'")

            # agent initialisation
            agents = COMA(agent_num, obs_dim, action_dim, lr_c, lr_a, gamma, target_update_steps, num_s)

            episode_reward = mp.Value('d', 0)
            average_traveling_time = mp.Value('d', 0)

            episodes_reward = mp.Queue()
            performance_list = mp.Queue()
            # training loop
            episode = mp.Value('i', 1)
            e = mp.Value('i', 0)

            workers = [
                Worker(agents, episode_reward, episodes_reward, episode, i, average_traveling_time,
                       performance_list, e, num_p, num_s, k, min_time)
                for i in range(8 - num_p, 8)]

            [w.start() for w in workers]
            retrun = []  # record episode reward to plot
            time_ = []  # record episode reward to plot
            while True:
                r = episodes_reward.get()
                t = performance_list.get()
                if r is not None:
                    retrun.append(r)
                    time_.append(t)
                else:

                    # data1为list类型，参数index为索引，column为列名
                    data1 = pd.DataFrame(data=retrun)
                    data2 = pd.DataFrame(data=time_)
                    df1[f"{num_s}_{num_p}_{k}"] = data1
                    df2[f"{num_s}_{num_p}_{k}"] = data2

                    end = time.time()
                    prediction = (end - start) / count * (30 - count) + end
                    localtime = time.asctime(time.localtime(prediction))
                    print(f'预计结束时间：{localtime}')

                    break
            [w.join() for w in workers]

    # PATH为导出文件的路径和文件名
    PATH1 = 'results/coma_return_train.csv'
    PATH2 = 'results/coma_time_train.csv'
    df1.to_csv(PATH1)
    df2.to_csv(PATH2)

    print('---training over---')
