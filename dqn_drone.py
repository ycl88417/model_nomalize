import setup_path
import gym
import airgym
import time

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
from typing import Callable



# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-sample-v0",
                ip_address="127.0.0.1",
                step_length=3,
                image_shape=(128,72,3),
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride =2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride = 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),         
            nn.MaxPool2d(kernel_size=2,stride = 2),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),         
            nn.MaxPool2d(kernel_size=2,stride = 2),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),  
            nn.Flatten(),
        )
        # calculate the shape of the flattened output of the CNN
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space['image'].sample()[None]).float()).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + observation_space['position'].shape[0] + observation_space['velocity'].shape[0]+ observation_space['waypoint'].shape[0], features_dim),
            nn.ReLU(),
        )

    def forward(self, observations_space: th.Tensor):
        cnn_features = self.cnn(observations_space['image'].float() / 255.0)
        other_features = th.cat((observations_space['position'], observations_space['velocity']),observations_space['waypoint'],dim=1)
        features = th.cat((cnn_features, other_features), dim=1)
        return self.linear(features)

policy_kwargs = dict(features_extractor_class=CustomCNN,
                    net_arch=[256,128,64,32,16]
                    )





# Initialize RL algorithm type and parameters
model = DQN(
    "MultiInputPolicy",
    env,
    learning_rate=0.025,
    gamma=0.9,
    verbose=1,
    batch_size=32,
    train_freq=4,
    target_update_interval=10000,
    learning_starts=20000,
    buffer_size=200000,
    max_grad_norm=10,
    exploration_fraction=0.8,
    exploration_final_eps=0.03,
    exploration_initial_eps=1,
    device="cuda",
    tensorboard_log="./tb_logs",
    policy_kwargs=policy_kwargs
)

print(model)
# Create an evaluation callback with the same env, called every 5000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=10000,
)
callbacks.append(eval_callback)
kwargs = {}
kwargs["callback"] = callbacks

models_dir = "."
timesteps = 10000
for i in range(50):
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False,**kwargs)
    model.save(f"{models_dir}/{timesteps*i}")