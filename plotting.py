import matplotlib.pyplot as plt
import torch
import numpy as np
import imageio
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init

# plot results
def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n

total_training_reward = torch.load('trainingResults/training_Reward.pt')

x_val = range(8192, (len(total_training_reward)+1)*8192, 8192)

plt.figure(figsize=(16,6))
plt.plot(x_val, total_training_reward, label='total training reward')
plt.plot(x_val,moving_average(total_training_reward), label = 'moving average')
plt.xlabel('time steps'); plt.ylabel('reward')
plt.xlim((0, max(x_val)*1.05))
plt.legend(loc=4); plt.grid()
plt.tight_layout(); plt.show()



"""Below cell can be used for policy evaluation and saves an episode to mp4 for you to view."""

env = make_env(n_envs=num_envs,env_name='coinrun',num_levels=num_levels)
encoder = Encoder(in_channels=3, feature_dim=4096)
policy = Policy(encoder=encoder, feature_dim=4096, num_actions=env.action_space.n)
policy.cuda()
policy.load_state_dict(torch.load('checkpoints/checkpoint.pt'))

# Make evaluation environment
eval_env = make_env(num_envs, start_level=num_levels, num_levels=num_levels, env_name='coinrun')
obs = eval_env.reset()

frames = []
total_reward = []
val_reward = []

# Evaluate policy
policy.eval()
for _ in range(512):

  # Use policy
  action, log_prob, value = policy.act(obs)

  # Take step in environment
  obs, reward, done, info = eval_env.step(action)
  val_reward.append(torch.Tensor(reward))

  # Render environment and store
  frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
  frames.append(frame)

# Calculate average return
total_reward = torch.stack(val_reward).sum(0).mean(0)
print('Average return:', total_reward)

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave('videos/vid2.mp4', frames, fps=25)