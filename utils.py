import xml.etree.ElementTree as et

import numpy as np
import pandas as pd


def get_average_travel_time():
    xtree = et.parse("scenario/sample.tripinfo.xml")
    xroot = xtree.getroot()

    rows = []
    for node in xroot:
        travel_time = node.attrib.get("duration")
        rows.append({"travel_time": travel_time})

    columns = ["travel_time"]
    travel_time = pd.DataFrame(rows, columns=columns).astype("float64")
    return travel_time["travel_time"].mean()


def moving_average(x, N):
    return np.convolve(x, np.ones((N,)) / N, mode='valid')


def record(episode, episode_reward, ep_r, episodes_reward, average_traveling_time, performance_list, name, e):
    with episode.get_lock():
        episode.value += 1

    if name == 'w07':
        with e.get_lock():
            e.value += 1

        with episode_reward.get_lock():
            episode_reward.value = ep_r

        with average_traveling_time.get_lock():
            average_traveling_time.value = get_average_travel_time()

        episodes_reward.put(episode_reward.value)
        performance_list.put(average_traveling_time.value)

        print(
            f"Episode:{e.value}\t Average Traveling Time:{average_traveling_time.value}\t episode_reward:{episode_reward.value}")
