# MCTS
Implementation of SPW and DPW for Monte Carlo Tree Search in Continuous action/state space

## Usage
Minimal example:

```python
import gym
from gym import spaces
import numpy as np
from mcts.MCTS import MCTS
from mcts.DPW import DPW
from mcts.SPW import SPW


class ContinousJump(gym.Env):
    """
    From Curtoux 2014 Monte Carlo Tree Search for Continuous and Stochastic Sequential Decision Making Problems
    """
    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self):
        self.action_space = spaces.Box(0,1,shape=[1])
        self.observation_space = spaces.Box(0,2,shape=[2])
        self.t_max = 2
        
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, t = self.state
        dx = action[0]
        
        noise = np.random.uniform()*0.05-0.025

        done =  t == 1
        
        x = x + dx + noise
        
        self.state = [x, t+1]

        if x >= 0 and x <= 1:
            reward = 0.7
        elif x >= 1.7:
            reward = 1
        else:
            reward = 0

        return np.array(self.state), reward, done, {}
    
    def seed(self, seed):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        np.random.seed(seed)

    def reset(self):
        self.state = [0,0]
        return np.array(self.state)

env = ContinousJump()
env.seed(2)
obs = env.reset()
model = DPW(alpha=0.3, beta=0.2, initial_obs=obs, env=env, K=3**0.5)
done = False

while not done:
    model.learn(10000, progress_bar=True)
    action = model.best_action()
    observation, reward, done, info = env.step(action)
    model.forward(action, observation)
    print("reward: {}\nnew state: {}".format(reward, np.round(observation[0],2)))
    if done:
        break
env.close()
```
