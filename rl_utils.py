import torch
from copy import deepcopy
#Visualise agent function
def visualise_agent(env, policy, n=5):
    try:
        for trial_i in range(n):
            observation = env.reset()
            done=False
            t=0
            episode_return=0
            while not done:
                env.render()
                action = policy(torch.tensor([observation]).double())
                observation, reward, done, info = env.step(action)
                episode_return+=reward
                t+=1
            env.render()
            time.sleep(1.5)
            print("Episode {} finished after {} timesteps. Return = {}".format(trial_i, t, episode_return))
        env.close()
    except KeyboardInterrupt:
        env.close()

### Upside Down RL ###
#Visualise agent
def visualise_agent_command(env, policy, command, n=5):
    try:
        for trial_i in range(n):
            current_command = deepcopy(command)
            observation = env.reset()
            done=False
            t=0
            episode_return=0
            while not done:
                env.render()
                action = policy(torch.tensor([observation]).double(), torch.tensor([command]).double())
                observation, reward, done, info = env.step(action)
                episode_return+=reward
                current_command[0]-= reward
                current_command[1] = max(1, current_command[1]-1)
                t+=1
            env.render()
            time.sleep(1.5)
            print("Episode {} finished after {} timesteps. Return = {}".format(trial_i, t, episode_return))
        env.close()
    except KeyboardInterrupt:
        env.close()

def create_greedy_policy(policy_network, command=False):
    if command:
        def policy(obs, command):
            action_logits = policy_network(obs, command)
            action = np.argmax(action_logits.detach().numpy())
            return action
    else:
        def policy(obs):
            action_logits = policy_network(obs)
            action = np.argmax(action_logits.detach().numpy())
            return action
    return policy

def create_stochastic_policy(policy_network, command=False):
    if command:
        def policy(obs, command):
            action_logits = policy_network(obs, command)
            action_probs = F.softmax(action_logits, dim=-1)
            action = torch.distributions.Categorical(action_probs).sample().item()
            return action
    else:
        def policy(obs):
            action_logits = policy_network(obs)
            action_probs = F.softmax(action_logits, dim=-1)
            action = torch.distributions.Categorical(action_probs).sample().item()
            return action
    return policy
