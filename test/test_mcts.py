import unittest
from mcts.MCTS import MCTS
from mcts.SPW import SPW
from mcts.DPW import DPW
import gym


class TestDeterministicTransition(unittest.TestCase):

    def setUp(self):
        self.n_sim = 10
        self.env = gym.make("FrozenLake-v0", is_slippery=False)
        initial_obs = self.env.reset()
        self.model_mcts = MCTS(initial_obs, self.env, K=2**0.5)
        self.model_spw = SPW(alpha=0.5, initial_obs=initial_obs, env=self.env, K=2**0.5)
        self.model_dpw = DPW(alpha=0.5, beta=0.5, initial_obs=initial_obs, env=self.env, K=2**0.5)

    def tearDown(self):
        pass

    def test_MCTS_detrministic(self):
        self.model_mcts.learn(self.n_sim)
        a = self.model_mcts.act()
        s, _, _, _ = self.env.step(a)
        self.model_mcts.forward(a, s)
        self.assertEqual(self.model_mcts.root.state, s)


class TestDiscreteActionsDiscreteState(unittest.TestCase):

    def setUp(self):
        self.n_sim = 10
        env = gym.make("FrozenLake8x8-v0")
        initial_obs = env.reset()
        self.model_mcts = MCTS(initial_obs, env, K=2**0.5)
        self.model_spw = SPW(alpha=0.5, initial_obs=initial_obs, env=env, K=2**0.5)
        self.model_dpw = DPW(alpha=0.5, beta=0.5, initial_obs=initial_obs, env=env, K=2**0.5)

    def tearDown(self):
        pass

    def test_MCTS_discrete_actions_discrete_state(self):
        self.model_mcts.learn(self.n_sim)

    def test_SPW_discrete_actions_discrete_state(self):
        self.model_spw.learn(self.n_sim)

    def test_DPW_discrete_actions_discrete_state(self):
        self.model_dpw.learn(self.n_sim)


class TestDiscreteActionsContinousState(unittest.TestCase):

    def setUp(self):
        self.n_sim = 10
        env = gym.make("CartPole-v0").env
        initial_obs = env.reset()
        self.model_mcts = MCTS(initial_obs, env, K=2**0.5)
        self.model_spw = SPW(alpha=0.5, initial_obs=initial_obs, env=env, K=2**0.5)
        self.model_dpw = DPW(alpha=0.5, beta=0.5, initial_obs=initial_obs, env=env, K=2**0.5)

    def tearDown(self):
        pass

    def test_MCTS_discrete_actions_continous_state(self):
        self.model_mcts.learn(self.n_sim)

    def test_SPW_discrete_actions_continous_state(self):
        self.model_spw.learn(self.n_sim)

    def test_DPW_discrete_actions_continous_state(self):
        self.model_dpw.learn(self.n_sim)


class TestContinousActionsContinousState(unittest.TestCase):

    def setUp(self):
        self.n_sim = 10
        self.env = gym.make("LunarLanderContinuous-v2").env
        self.initial_obs = self.env.reset()

    def tearDown(self):
        pass

    def test_MCTS_continuous_actions_continous_state(self):
        self.assertRaises(Exception, MCTS.__init__, initial_obs=self.initial_obs, env=self.env, K=2**0.5)

    def test_SPW_continuous_actions_continous_state(self):
        self.model_spw = SPW(alpha=0.5, initial_obs=self.initial_obs, env=self.env, K=2**0.5)
        self.model_spw.learn(self.n_sim)

    def test_DPW_continuous_actions_continous_state(self):
        self.model_dpw = DPW(alpha=0.5, beta=0.5, initial_obs=self.initial_obs, env=self.env, K=2**0.5)
        self.model_dpw.learn(self.n_sim)


if __name__ == "__main__":
    unittest.main()
