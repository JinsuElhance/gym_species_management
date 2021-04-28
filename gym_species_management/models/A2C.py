from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C
import gym
import ipdb

import gym_species_management

ipdb.set_trace()
env = make_vec_env("species_management-v0", n_envs=4)
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("a2c_species_management")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()