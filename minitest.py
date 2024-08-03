# from dm_control import suite
# import gym
#
# gym_env = gym.make("Hopper-v2")
# dm_env = suite.load(domain_name="hopper", task_name="stand")
# # dm_env.physics.speed()
# body_names = gym_env.env.model.body_names
# body_idxes = {name: idx for idx, name in enumerate(body_names)}
# print(body_names)
# print(body_idxes)
#
# for i in range(1, 5):
#     print(gym_env.env.sim.data.body_xpos[i])

import json
from omegaconf import OmegaConf, DictConfig
import numpy as np


def main():
    env_cfg = OmegaConf.create(json.load(open("config/config.json", "r")))
    from multitask_mujoco import MultitaskMujoco

    env = MultitaskMujoco(env_cfg)
    for i, mini_env in enumerate(env.envs):
        print(env.tasks[i])
        print(env.envs_info[i])
    n_episodes = 13

    for e in range(n_episodes):



        env.reset_task(e % len(env.tasks))
        print(env.task)
        env.reset()
        terminated = [False]
        episode_reward = 0

        while not all(terminated):
            # obs_n = env.reset()
            # obs = env.get_obs()
            # state = env.get_state()

            actions = []
            for agent_id in range(env.n_agents):
                # avail_actions = env.get_avail_agent_actions(agent_id)
                # avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.uniform(-1.0, 1.0, env.action_shape)
                actions.append(action)

            # obs_n, reward_n, terminated, info = env.step(actions)
            obs_n, share_obs_n, reward_n, terminated, info_n, avail_actions = env.step(actions)
            # print(env.steps)

            episode_reward += reward_n[0]

            # time.sleep(0.1)
            # print("reward_n = {}".format(reward_n))
            # env.render()


        # print("Total reward in episode {} = {}".format(e, episode_reward))

    env.close()

if __name__ == "__main__":
    main()










