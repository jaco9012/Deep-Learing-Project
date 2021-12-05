import matplotlib.pyplot as plt
import torch
import numpy as np
import imageio
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init
from math import sqrt, exp
from random import random, sample

# Models to make videos for
savename_baseline="baseline_v4"
savename_IMPALA="IMPALA_v4"
savename_IMPALA_rand_conv="IMPALA_rand_conv_v4"

# Hyperparameters
num_envs = 64
num_levels = 0 # 0 = unlimited levels


# plot results
def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

#### BASELINE ####

class Encoder(nn.Module):
  def __init__(self, in_channels, feature_dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=1024, out_features=feature_dim), nn.ReLU()
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    return self.layers(x)

class Policy(nn.Module):
  def __init__(self, encoder, feature_dim, num_actions):
    super().__init__()
    self.encoder = encoder
    self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
    self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)

  def act(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value = self.forward(x)
      action = dist.sample()
      log_prob = dist.log_prob(action)    
    return action.cpu(), log_prob.cpu(), value.cpu()

  def act_greedy(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value = self.forward(x)
      action = torch.argmax(dist.probs,dim=1)
      log_prob = dist.log_prob(action)
    return action.cpu(), log_prob.cpu(), value.cpu()

  def select_act(self, x, eps_end, eps_start, eps_decay, step):
    sample = random()
    eps_threshold = eps_end + (eps_start - eps_end) * exp(-1 * step / eps_decay)
    if sample > eps_threshold:
      return self.act_greedy(x) 
    else:
      return self.act(x)

  def forward(self, x):
    x = self.encoder(x)
    logits = self.policy(x)
    value = self.value(x).squeeze(1)
    dist = torch.distributions.Categorical(logits=logits)

    return dist, value

# Make evaluation environment
eval_env = make_env(num_envs, start_level=num_levels, num_levels=0, env_name='coinrun')
eval_obs = eval_env.reset()

# Define network
encoder = Encoder(in_channels=3, feature_dim=4096)
policy = Policy(encoder=encoder, feature_dim=4096, num_actions=eval_env.action_space.n)
policy.cuda()
policy.load_state_dict(torch.load('checkpoints/' + savename_baseline + '.pt'))

frames = []
total_reward = []
val_reward = []

# Evaluate policy
policy.eval()
for _ in range(512):

  # Use policy
  action, log_prob, value = policy.act(eval_obs)

  # Take step in environment
  eval_obs, reward, done, info = eval_env.step(action)
  val_reward.append(torch.Tensor(reward))

  # Render environment and store
  frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
  frames.append(frame)

# Calculate average return
total_reward = torch.stack(val_reward).sum(0).mean(0)
print('Average return baseline:', total_reward)

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave('videos/' + savename_baseline + '.mp4', frames, fps=25)




#### IMPALA ####



class Encoder(nn.Module):
  def __init__(self, in_channels, feature_dim):
    super().__init__()
    self.layers = nn.Sequential(
        # outchannels 16
        nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.ReLU(), 
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(), 
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),

        # outchannels 32
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.ReLU(), 
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(), 
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),

        # outchannels 32
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.ReLU(), 
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(), 
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=2048, out_features=feature_dim), nn.ReLU()
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    return self.layers(x)

class Policy(nn.Module):
  def __init__(self, encoder, feature_dim, num_actions):
    super().__init__()
    self.encoder = encoder
    self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
    self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)

  def act(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value = self.forward(x)
      action = dist.sample()
      log_prob = dist.log_prob(action)    
    return action.cpu(), log_prob.cpu(), value.cpu()

  def act_greedy(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value = self.forward(x)
      action = torch.argmax(dist.probs,dim=1)
      log_prob = dist.log_prob(action)
    return action.cpu(), log_prob.cpu(), value.cpu()

  def select_act(self, x, eps_end, eps_start, eps_decay, step):
    sample = random()
    eps_threshold = eps_end + (eps_start - eps_end) * exp(-1 * step / eps_decay)
    if sample > eps_threshold:
      return self.act_greedy(x) 
    else:
      return self.act(x)

  def forward(self, x):
    x = self.encoder(x)
    logits = self.policy(x)
    value = self.value(x).squeeze(1)
    dist = torch.distributions.Categorical(logits=logits)

    return dist, value


# Make evaluation environment
eval_env = make_env(num_envs, start_level=num_levels, num_levels=0, env_name='coinrun')
eval_obs = eval_env.reset()

# Define network
encoder = Encoder(in_channels=3, feature_dim=256)
policy = Policy(encoder=encoder, feature_dim=256, num_actions=eval_env.action_space.n)
policy.cuda()
policy.load_state_dict(torch.load('checkpoints/' + savename_IMPALA + '.pt'))

frames = []
total_reward = []
val_reward = []

# Evaluate policy
policy.eval()
for _ in range(512):

  # Use policy
  action, log_prob, value = policy.act(eval_obs)

  # Take step in environment
  eval_obs, reward, done, info = eval_env.step(action)
  val_reward.append(torch.Tensor(reward))

  # Render environment and store
  frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
  frames.append(frame)

# Calculate average return
total_reward = torch.stack(val_reward).sum(0).mean(0)
print('Average return IMPALA:', total_reward)

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave('videos/' + savename_IMPALA + '.mp4', frames, fps=25)




#### IMPALA RAND CONV ####




class Encoder(nn.Module):
  def __init__(self, in_channels, feature_dim):
    super().__init__()
    self.layers = nn.Sequential(
        # outchannels 16
        nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.ReLU(), 
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(), 
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),

        # outchannels 32
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.ReLU(), 
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(), 
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),

        # outchannels 32
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.ReLU(), 
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(), 
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=2048, out_features=feature_dim), nn.ReLU()
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    return self.layers(x)

class Policy(nn.Module):
  def __init__(self, encoder, feature_dim, num_actions):
    super().__init__()
    self.encoder = encoder
    self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
    self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)

  def act(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value = self.forward(x)
      action = dist.sample()
      log_prob = dist.log_prob(action)    
    return action.cpu(), log_prob.cpu(), value.cpu()

  def act_greedy(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value = self.forward(x)
      action = torch.argmax(dist.probs,dim=1)
      log_prob = dist.log_prob(action)
    return action.cpu(), log_prob.cpu(), value.cpu()

  def select_act(self, x, eps_end, eps_start, eps_decay, step):
    sample = random()
    eps_threshold = eps_end + (eps_start - eps_end) * exp(-1 * step / eps_decay)
    if sample > eps_threshold:
      return self.act_greedy(x) 
    else:
      return self.act(x)

  def forward(self, x):
    x = self.encoder(x)
    logits = self.policy(x)
    value = self.value(x).squeeze(1)
    dist = torch.distributions.Categorical(logits=logits)

    return dist, value

class RandConv(nn.Module):
  def __init__(self, num_batch):
    super().__init__()
    
    self.randconv = nn.Conv2d(3, 3, kernel_size=3, bias=False, padding=1)
    torch.nn.init.xavier_normal_(self.randconv.weight.data)

  def RandomConvolution(self, imgs):
    _device = imgs.device
    self.randconv.to(_device)
    img_h, img_w = imgs.shape[2], imgs.shape[3]
    num_stack_channel = imgs.shape[1]
    num_batch = imgs.shape[0]
    num_trans = num_batch
    batch_size = int(num_batch / num_trans)

    for trans_index in range(num_trans):
        temp_imgs = imgs[trans_index*batch_size:(trans_index+1)*batch_size]
        temp_imgs = temp_imgs.reshape(-1, 3, img_h, img_w) # (batch x stack, channel, h, w)
        with torch.no_grad():
            rand_out = self.randconv(temp_imgs)
        if trans_index == 0:
            total_out = rand_out
        else:
            total_out = torch.cat((total_out, rand_out), 0)
    total_out = total_out.reshape(-1, num_stack_channel, img_h, img_w)
    return total_out


# Make evaluation environment
eval_env = make_env(num_envs, start_level=num_levels, num_levels=0, env_name='coinrun')
eval_obs = eval_env.reset()

# Define network
encoder = Encoder(in_channels=3, feature_dim=256)
policy = Policy(encoder=encoder, feature_dim=256, num_actions=eval_env.action_space.n)
policy.cuda()
policy.load_state_dict(torch.load('checkpoints/' + savename_IMPALA_rand_conv + '.pt'))

frames = []
total_reward = []
val_reward = []

# Evaluate policy
policy.eval()
for _ in range(512):

  # Use policy
  action, log_prob, value = policy.act(eval_obs)

  # Take step in environment
  eval_obs, reward, done, info = eval_env.step(action)
  val_reward.append(torch.Tensor(reward))

  # Render environment and store
  frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
  frames.append(frame)

# Calculate average return
total_reward = torch.stack(val_reward).sum(0).mean(0)
print('Average return IMPALA rand conv:', total_reward)

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave('videos/' + savename_IMPALA_rand_conv + '.mp4', frames, fps=25)




