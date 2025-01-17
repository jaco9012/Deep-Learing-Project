import matplotlib.pyplot as plt
import torch
import numpy as np
import imageio
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init


# Models to make plots for
savename_baseline="baseline_v6"
savename_IMPALA="IMPALA_v6"
savename_IMPALA_rand_conv="IMPALA_rand_conv_v8"
savename_results="resultsv7"

# plot results
def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n

total_training_reward_baseline = torch.load('trainingResults/training_Reward_' + savename_baseline + '.pt')
total_training_reward_IMPALA = torch.load('trainingResults/training_Reward_' + savename_IMPALA + '.pt')
total_training_reward_rand_conv = torch.load('trainingResults/training_Reward_' + savename_IMPALA_rand_conv + '.pt')

x_train_baseline = range(0, (len(total_training_reward_baseline))*8192*2, 8192*2)
x_train_IMPALA = range(0, (len(total_training_reward_IMPALA))*8192*2, 8192*2)
x_train_rand_conv = range(0, (len(total_training_reward_rand_conv))*8192*2, 8192*2)

total_validation_reward_baseline = torch.load('trainingResults/validation_Reward_' + savename_baseline + '.pt')
total_validation_reward_IMPALA = torch.load('trainingResults/validation_Reward_' + savename_IMPALA + '.pt')
total_validation_reward_rand_conv = torch.load('trainingResults/validation_Reward_' + savename_IMPALA_rand_conv + '.pt')

x_val_baseline = range(0, (len(total_validation_reward_baseline))*196608, 196608)
x_val_IMPALA = range(0, (len(total_validation_reward_IMPALA))*196608, 196608)
x_val_rand_conv = range(0, (len(total_validation_reward_rand_conv))*196608, 196608)

plt.subplots(nrows=3,ncols=1,sharex='col', figsize=(12,6))
# Baseline
plt.subplot(3,1,1)
plt.plot(x_train_baseline, total_training_reward_baseline, label='Total Training Reward')
plt.plot(x_train_baseline,moving_average(total_training_reward_baseline), label = 'Moving Average')
plt.plot(x_val_baseline, total_validation_reward_baseline, label='Total Validation Reward')
plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.3), ncol=3, fancybox=True)
plt.title(label='Nature CNN', loc='left')
plt.grid()
# IMPALA
plt.subplot(3,1,2)
plt.plot(x_train_IMPALA, total_training_reward_IMPALA)
plt.plot(x_train_IMPALA,moving_average(total_training_reward_IMPALA))
plt.plot(x_val_IMPALA, total_validation_reward_IMPALA)
plt.ylabel('Average Reward')
plt.grid()
plt.title(label='IMPALA CNN', loc='left')
# Rand Conv
plt.subplot(3,1,3)
plt.plot(x_train_rand_conv, total_training_reward_rand_conv)
plt.plot(x_train_rand_conv,moving_average(total_training_reward_rand_conv))
plt.plot(x_val_rand_conv, total_validation_reward_rand_conv)
plt.xlabel('Time Steps')
plt.title(label='IMPALA CNN + Rand. Conv.', loc='left')
plt.xlim((0, max(x_train_baseline)*1.02))
plt.grid()
plt.tight_layout(h_pad=1.1)
plt.savefig('videos/'+ savename_results + '.png', transparent=True)
plt.show()






plt.figure(figsize=(16,6))
plt.plot(x_train_baseline, total_training_reward_baseline, label='total training reward baseline')
plt.plot(x_train_baseline,moving_average(total_training_reward_baseline), label = 'moving average baseline')
plt.plot(x_train_IMPALA, total_training_reward_IMPALA, label='total training reward IMPALA')
plt.plot(x_train_IMPALA,moving_average(total_training_reward_IMPALA), label = 'moving average IMPALA')
plt.plot(x_train_rand_conv, total_training_reward_rand_conv, label='total training reward rand conv')
plt.plot(x_train_rand_conv,moving_average(total_training_reward_rand_conv), label = 'moving average rand conv')
plt.xlabel('time steps'); plt.ylabel('reward')
plt.xlim((0, max(x_train_baseline)*1.05))
plt.legend(loc=0); plt.grid()
plt.tight_layout(); plt.show()


total_validation_reward_baseline = torch.load('trainingResults/validation_Reward_' + savename_baseline + '.pt')
total_validation_reward_IMPALA = torch.load('trainingResults/validation_Reward_' + savename_IMPALA + '.pt')
total_validation_reward_rand_conv = torch.load('trainingResults/validation_Reward_' + savename_IMPALA_rand_conv + '.pt')

x_val_baseline = range(8192*2, (len(total_validation_reward_baseline)+1)*8192*2, 8192*2)
x_val_IMPALA = range(8192*2, (len(total_validation_reward_IMPALA)+1)*8192*2, 8192*2)
x_val_rand_conv = range(8192*2, (len(total_validation_reward_rand_conv)+1)*8192*2, 8192*2)

plt.figure(figsize=(16,6))
plt.plot(x_val_baseline, total_validation_reward_baseline, label='total validation reward baseline')
#plt.plot(x_val_baseline,moving_average(total_validation_reward_baseline), label = 'moving average baseline')
plt.plot(x_val_IMPALA, total_validation_reward_IMPALA, label='total validation reward IMPALA')
#plt.plot(x_val_IMPALA,moving_average(total_validation_reward_IMPALA), label = 'moving average IMPALA')
plt.plot(x_val_rand_conv, total_validation_reward_rand_conv, label='total validation reward rand conv')
#plt.plot(x_val_rand_conv,moving_average(total_validation_reward_rand_conv), label = 'moving average rand conv')
plt.xlabel('time steps'); plt.ylabel('reward')
plt.xlim((0, max(x_val_baseline)*1.05))
plt.legend(loc=0); plt.grid()
plt.tight_layout(); plt.show()
