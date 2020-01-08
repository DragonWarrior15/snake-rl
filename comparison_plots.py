import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_base = pd.read_csv('model_logs/v15.1.csv')
df_batch = pd.read_csv('model_logs/v15.4.csv')
df_super = pd.read_csv('model_logs/v15.2.csv')
df_reward = pd.read_csv('model_logs/v15.3.csv')

df_base['length_mean_ma'] = df_base['length_mean'].rolling(10).mean()
df_batch['length_mean_ma'] = df_batch['length_mean'].rolling(10).mean()
df_super['length_mean_ma'] = df_super['length_mean'].rolling(10).mean()
df_reward['length_mean_ma'] = df_reward['length_mean'].rolling(10).mean()
df_base['loss_ma'] = df_base['loss'].rolling(10).mean()
df_batch['loss_ma'] = df_batch['loss'].rolling(10).mean()
df_super['loss_ma'] = df_super['loss'].rolling(10).mean()
df_reward['loss_ma'] = df_reward['loss'].rolling(10).mean()

fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.set_title('Snake Mean Length vs Training Batch Size')

axs.plot(df_base['iteration'][:200], df_base['length_mean'][:200], 
        label='Batch Size 64', color='skyblue')
axs.plot(df_batch['iteration'][:200], df_batch['length_mean'][:200], 
        label='Batch Size 128', color='bisque')

axs.plot(df_base['iteration'][9:200], df_base['length_mean_ma'][9:200], 
        label='Batch Size 64 Moving Average', color='blue')
axs.plot(df_batch['iteration'][9:200], df_batch['length_mean_ma'][9:200], 
        label='Batch Size 128 Moving Average', color='red')

axs.set_ylabel('Mean Length')
axs.set_xlabel('Iteration')

plt.legend()
plt.show()


fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.set_title('Snake Mean Length vs PreTraining')

axs.plot(df_base['iteration'][:100], df_base['length_mean'][:100], 
        label='DQN', color='skyblue')
axs.plot(df_super['iteration'][:100], df_super['length_mean'][:100], 
        label='DQN PreTrained', color='bisque')

axs.plot(df_base['iteration'][9:100], df_base['length_mean_ma'][9:100], 
        label='DQN Moving Average', color='blue')
axs.plot(df_super['iteration'][9:100], df_super['length_mean_ma'][9:100], 
        label='DQN PreTrained Moving Average', color='red')

axs.set_ylabel('Mean Length')
axs.set_xlabel('Iteration')

plt.legend()
plt.show()


fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.set_title('Snake Mean Length vs Reward Type')

axs.plot(df_base['iteration'][:100], df_base['length_mean'][:100], 
        label='Static Reward', color='skyblue')
axs.plot(df_reward['iteration'][:100], df_reward['length_mean'][:100], 
        label='Length Dependent Reward', color='bisque')

axs.plot(df_base['iteration'][9:100], df_base['length_mean_ma'][9:100], 
        label='Static Reward Moving Average', color='blue')
axs.plot(df_reward['iteration'][9:100], df_reward['length_mean_ma'][9:100], 
        label='Length Dependent Reward Moving Average', color='red')

axs.set_ylabel('Mean Length')
axs.set_xlabel('Iteration')

plt.legend()
plt.show()


fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.set_title('Snake Mean Length vs Reward Type')

axs.plot(df_base['iteration'][:100], df_base['loss'][:100], 
        label='Static Reward', color='skyblue')
axs.plot(df_reward['iteration'][:100], df_reward['loss'][:100], 
        label='Length Dependent Reward', color='bisque')

axs.plot(df_base['iteration'][9:100], df_base['loss_ma'][9:100], 
        label='Static Reward Moving Average', color='blue')
axs.plot(df_reward['iteration'][9:100], df_reward['loss_ma'][9:100], 
        label='Length Dependent Reward Moving Average', color='red')

axs.set_ylabel('Mean Length')
axs.set_xlabel('Iteration')

plt.legend()
plt.show()
