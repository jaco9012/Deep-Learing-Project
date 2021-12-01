# -*- coding: utf-8 -*-

# Network definitions. 
# We have defined a policy network for you in advance. It uses the popular `NatureDQN` encoder architecture (see below),
# while policy and value functions are linear projections from the encodings. There is plenty of opportunity to experiment with architectures,
# so feel free to do that! Perhaps implement the `Impala` encoder from [this paper](https://arxiv.org/pdf/1802.01561.pdf) (perhaps minus the LSTM).


from math import sqrt, exp
from random import random, sample
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init
from labml_nn.rl.ppo import ClippedPPOLoss, ClippedValueFunctionLoss



# Hyperparameters
total_steps = 20e6
num_envs = 64
num_levels = 0 # 0 = unlimited levels
num_steps = 256
num_epochs = 3
batch_size = 512
eps = .2
eps_end = 0.05
eps_start = 0.9
eps_decay = 10e6
grad_eps = .5
clip_value = .2
value_coef = .5
entropy_coef = .01
gamma = 0.999



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

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

  def act_gready(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value = self.forward(x)
      action = torch.argmax(dist)
      log_prob = dist.log_prob(action)
    return action.cpu(), log_prob.cpu(), value.cpu()

  def select_act(self, x, eps_end, eps_start, eps_decay, step):
    sample = random()
    eps_threshold = eps_end + (eps_start - eps_end) * exp(-1 * step / eps_decay)
    if sample > eps_threshold:
      return self.act_gready(x) 
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

# Define environment
# check the utils.py file for info on arguments
env = make_env(n_envs=num_envs,env_name='coinrun',num_levels=num_levels)
print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)


# Define network
encoder = Encoder(in_channels=3, feature_dim=256)
policy = Policy(encoder=encoder, feature_dim=256, num_actions=env.action_space.n)
policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4*1/sqrt(32/3), eps=1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs,
    gamma = gamma
)

clipped_PPO_loss = ClippedPPOLoss()
clipped_value_loss = ClippedValueFunctionLoss()

# Run training
obs = env.reset()
step = 0
total_training_reward = []
total_val_reward = []

augmentation="rand_conv"

while step < total_steps:
  randConvGenerator = RandConv(num_batch=64)
  val_reward = []
  # Use policy to collect data for num_steps steps
  policy.eval()
  for _ in range(num_steps):
    # apply data augmentation
    if augmentation == "rand_conv":
      obs = randConvGenerator.RandomConvolution(obs)
    # Use policy
    action, log_prob, value = policy.act(obs)
    
    # Take step in environment
    next_obs, reward, done, info = env.step(action)
    val_reward.append(torch.Tensor(reward))

    # Store data
    storage.store(obs, action, reward, done, info, log_prob, value)
    
    # Update current observation
    obs = next_obs

  # Add the last observation to collected data
  _, _, value = policy.act(obs)
  storage.store_last(obs, value)

  # Compute return and advantage
  storage.compute_return_advantage()

  # Optimize policy
  policy.train()
  for epoch in range(num_epochs):

    # Iterate over batches of transitions
    generator = storage.get_generator(batch_size)
    for batch in generator:
      b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

      # apply data augmentation
      if augmentation == "rand_conv":
        b_obs = randConvGenerator.RandomConvolution(b_obs)

      # Get current policy outputs
      new_dist, new_value = policy(b_obs)
      new_log_prob = new_dist.log_prob(b_action)

      # Clipped policy objective
      pi_loss = clipped_PPO_loss(log_pi=new_log_prob, sampled_log_pi=b_log_prob, advantage=b_advantage, clip=clip_value)
      
      # Clipped value function objective
      value_loss = clipped_value_loss(value=new_value, sampled_value=b_value, sampled_return=b_returns, clip=clip_value)

      # Entropy loss
      entropy_loss = new_dist.entropy()
      entropy_loss = entropy_loss.mean()

      # Backpropagate losses
      loss = (pi_loss + value_coef * value_loss - entropy_coef * entropy_loss) 
      loss.backward()

      # Clip gradients
      torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

      # Update policy
      optimizer.step()
      optimizer.zero_grad()

  # Update stats
  # total_training_reward.append(storage.get_reward())
  total_val_reward.append(torch.stack(val_reward).sum(0).mean(0))
  step += num_envs * num_steps

  if(step % 999424 == 0): # we save every 1e6 ish timesteps
    torch.save(policy.state_dict(), 'checkpoints/IMPALA_proc_rand_conv.pt')
    torch.save(total_val_reward, 'trainingResults/training_Reward_IMPALA_rand_conv.pt')

    

print('Completed training!')

torch.save(policy.state_dict(), 'checkpoints/IMPALA_proc_rand_conv.pt')
torch.save(total_val_reward, 'trainingResults/training_Reward_IMPALA_rand_conv.pt')
