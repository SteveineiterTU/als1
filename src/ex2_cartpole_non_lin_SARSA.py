import gym
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss
from torch.optim import Adam

from cartpole_utils import plot_results
from utils import env_reset, env_step


# TODO HIER WEITER :3
#  dann parameter anpassen, zb fuer alle sachen je 2 neue sachen, rest gleich lassen zB nhidden 10, 30...

seed = 42
env = gym.make("CartPole-v0")
env.seed(seed)
torch.manual_seed(seed)

num_actions = env.action_space.n
num_observations = env.observation_space.shape[0]
Q = torch.zeros([num_observations, num_actions])

# Parameters
num_hidden = 20  # 10, 40
alpha = 1e-3
eps = 1  # 0.01, 0.9
gamma = 0.9  # 0.2, 1
eps_decay = 0.99  # 0.1, 0.8
max_train_iterations = 1000
max_test_iterations = 100
max_episode_length = 200


def convert(x):
    return torch.tensor(x).float().unsqueeze(0)


model = nn.Sequential(
    nn.Linear(in_features=num_observations, out_features=num_hidden),
    nn.ReLU(),
    nn.Linear(in_features=num_hidden, out_features=num_actions)
)


def policy(state, is_training):
    """ epsilon-greedy policy
    """

    global eps

    with torch.no_grad():
        if is_training and np.random.uniform(0, 1) < eps:  # choose actions deterministically during test phase
            return torch.tensor(np.random.choice(num_actions))  # return a random action with probability epsilon
        else:
            return np.argmax(model(torch.tensor(state)))  # otherwise return the action approximated by linear model


criterion = MSELoss()  # using MSE instead of least squared error only differs in a scaling factor of 1/2 instead of 1/N for N samples which only affects convergence rate but not the global
# minimization
optimizer = Adam(model.parameters(), lr=alpha)


def compute_loss(state, action, reward, next_state, next_action, done):
    state = convert(state)
    next_state = convert(next_state)
    action = action.view(1, 1)
    next_action = next_action.view(1, 1)
    reward = torch.tensor(reward).view(1, 1)
    done = torch.tensor(done).int().view(1, 1)

    q = model(state)[0][action]
    with torch.no_grad():
        next_q = model(next_state)[0][next_action]
        if done:
            next_q = torch.zeros_like(next_q)

    output = q
    target = reward + gamma * next_q

    return criterion(output, target)


def train_step(state, action, reward, next_state, next_action, done):
    loss = compute_loss(state, action, reward, next_state, next_action, done)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def run_episode(is_training, is_rendering=False):
    global eps
    episode_reward, episode_loss = 0, 0.
    state = env_reset(env, is_rendering)
    action = policy(state, eps)
    for t in range(max_episode_length):
        next_state, reward, done, _ = env_step(env, action.item(), is_rendering)
        episode_reward += reward
        next_action = policy(next_state, is_training)

        if is_training:
            episode_loss += train_step(state, action, reward, next_state, next_action, done)
        else:
            with torch.no_grad():
                episode_loss += compute_loss(state, action, reward, next_state, next_action, done).item()

        state, action = next_state, next_action
        if done:
            break
    return dict(reward=episode_reward, loss=episode_loss / t)


def update_metrics(metrics, episode):
    for k, v in episode.items():
        metrics[k].append(v)


def print_metrics(it, metrics, is_training, window=100):
    reward_mean = np.mean(metrics['reward'][-window:])
    loss_mean = np.mean(metrics['loss'][-window:])
    mode = "train" if is_training else "test"
    print(f"It {it:4d} | {mode:5s} | reward {reward_mean:5.1f} | loss {loss_mean:5.2f}")


train_metrics = dict(reward=[], loss=[])
for it in range(max_train_iterations):
    episode_metrics = run_episode(is_training=True)
    update_metrics(train_metrics, episode_metrics)
    if it % 100 == 0:
        print_metrics(it, train_metrics, is_training=True)
    eps *= eps_decay

test_metrics = dict(reward=[], loss=[])
for it in range(max_test_iterations):
    episode_metrics = run_episode(is_training=False)
    update_metrics(test_metrics, episode_metrics)
    print_metrics(it + 1, test_metrics, is_training=False)
plot_results(train_metrics, test_metrics)

# you can visualize the trained policy like this:
# run_episode(is_training=False, is_rendering=True)
