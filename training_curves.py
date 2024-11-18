import pandas as pd
import matplotlib.pyplot as plt
import glob


log_files = glob.glob('./logs/train_*.monitor.csv')


all_rewards = []
all_timesteps = []


for log_file in log_files:
    
    data = pd.read_csv(log_file, skiprows=1)
    
    # Extract timesteps and rewards
    timesteps = data['t']
    rewards = data['r']
    
    # Accumulate rewards and timesteps
    all_timesteps.extend(timesteps)
    all_rewards.extend(rewards)

# Convert to cumulative timesteps for plotting
cumulative_timesteps = [sum(all_timesteps[:i+1]) for i in range(len(all_timesteps))]

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(cumulative_timesteps, all_rewards, label='Reward', color='blue', alpha=0.7)


plt.xlabel('Cumulative Timesteps')
plt.ylabel('Episode Reward')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)


plt.savefig('learning_curve.png')
plt.show()