import numpy as np
import pdb
from tqdm import tqdm_notebook as tqdm


class Decision_Node:
    """
    The Decision Noed class

    :param state: (tuple) defining the state
    :param father: (RandomNode) The father of the Decision Node, None if it is the root node
    :param is_root: (bool)
    :param is_final: (bool)
    """

    def __init__(self, state, father=None, is_root=False, is_final=False):
        self.state = state
        self.children = {}
        self.is_final = is_final
        self.visits = 0
        self.father = father
        self.is_root = is_root

    def add_children(self, random_node):
        """
        Adds a RandomNode  object to the dictionary of childrens (key is the action)

        :param random_node: (RandomNode) add a raindom node to the set of children visited
        """
        self.children[str(random_node.action)] = random_node

    def __repr__(self):
        s = ""
        for item in self.__dict__.items():
            if item[0] == "children":
                s += " "+"(children, "+str(len(self.children))+")"
            elif item[0] == "father":
                pass
            else:
                s += " "+str(item)
        return s


class RandomNode:
    """
    The RandomNode class defined by the state and action taken, is a ranodm node since the next state is note yet defined

    :param state: (tuple) defining the state
    :param action: (action) taken in the decision node
    :param father: (DecisionNode)
    :param is_root: (bool)
    :param is_final: (bool)
    """

    def __init__(self, state, action, father=None):
        self.state = state
        self.action = action
        self.children = {}
        self.cumulative_reward = 0
        self.visits = 0
        self.father = father

    def reward(self, reward):
        self.reward = reward

    def add_children(self, x):
        """
        adds a DecisionNode object to the dictionary of childrens (key is the state)

        :param x: (DecisinNode) the decision node to add to the children dict
        """
        self.children[str(x.state)] = x

    def __repr__(self):
        s = ""
        for item in self.__dict__.items():
            if item[0] == "children":
                s += " "+"(children, "+str(len(self.children))+")"
            elif item[0] == "father":
                pass
            else:
                s += " "+str(item)
        return s


class MCTS:
    """
    Base class for MCTS based on Monte Carlo Tree Search for Continuous and Stochastic Sequential Decision Making Problems, Courtoux

    :param initial_state: (DecisionNode) initial state of the tree
    :param K: exploration parameter of UCB
    :param generative_model: (fun: RandomNode -> [DecisionNode,float]) Function that gets a RandomNode and returns a DecisionNode and a float.
    :param rollout_policy: (fun: DecisionNode -> float) Function that selects an action from a DecisionNode during the rollout procedure
    :param action_sampler: (fun: DecisionNode -> float) Function that selects an action from a DecisionNode during the sampling procedure
    """

    def __init__(self, initial_state, K, generative_model, rollout_policy, action_sampler):
        self.root = initial_state
        self.K = K
        self.generative_model = generative_model
        self.rollout_policy = rollout_policy
        self.action_sampler = action_sampler

    def grow_tree(self):
        """
        Explores the current tree with the UCB principle untile we reach an unvisited node where the reward is obtained with random rollouts
        """

        decision_node = self.root

        while not (decision_node.is_final or decision_node.visits == 0):

            a = self.select(decision_node)
            x = decision_node.state

            if str(a) not in decision_node.children.keys():
                new_random_node = RandomNode(x, a, father=decision_node)
                decision_node.add_children(new_random_node)
            else:
                new_random_node = decision_node.children[str(a)]

            (new_decision_node, r) = self.select_outcome(new_random_node)

            if str(new_decision_node.state) not in new_random_node.children.keys():
                new_decision_node.father = new_random_node
                new_random_node.add_children(new_decision_node)
            else:
                new_decision_node = new_random_node.children[str(new_decision_node.state)]

            new_decision_node.reward = r
            new_random_node.reward = r

            decision_node = new_decision_node

        cumulative_reward = self.evaluate(decision_node)

        while not decision_node.is_root:
            random_node = decision_node.father
            cumulative_reward += random_node.reward
            random_node.cumulative_reward += cumulative_reward
            random_node.visits += 1
            decision_node = random_node.father
            decision_node.visits += 1

    def evaluate(self, x):
        """
        Evaluates a DecionNode playing until an terminal node using the rollotPolicy

        :param x: (Decision_Node) Node to evaluate
        :return: (float) the cumulative reward observed during the tree traversing.
        """

        R = 0
        x.visits += 1
        while not x.is_final:
            a = self.rollout_policy(x)
            new_random_node = RandomNode(x.state, a, father=x)
            (x, r) = self.generative_model(new_random_node)
            R += r

        return R

    def select_outcome(self, random_node):
        """
        Given a RandomNode returns a DecisionNode

        :param: random_node: (RandomNode) the random node from which selects the next state
        :return: (DecisionNode) the selected Decision Node
        """

        return self.generative_model(random_node)

    def select(self, x):
        """
        Selects the action to play from the current decision node

        :param x: (DecisionNode) current decision node
        :return: (float) action to play
        """

        raise NotImplemented

    def best_action(self):
        """
        At the end of the simulations returns the most visited action

        :return: (float) the best action accoring to the number of visits
        """

        number_of_visits_children = [node.visits for node in self.root.children.values()]
        index_best_action = np.argmax(number_of_visits_children)

        return list(self.root.children.values())[index_best_action].action

    def run(self, Nsim, progress_bar=False):
        """
        Expand the tree and return the bet action

        :param: Nsim: (int) number of tree traversals to do
        :return: (float) the best action selected
        """

        if progress_bar:
            iterations = tqdm(range(Nsim))
        else:
            iterations = range(Nsim)
        for _ in iterations:
            self.grow_tree()

        return self.best_action()


class SPW(MCTS):
    """
    Simple Progressive Widening trees based on Monte Carlo Tree Search for Continuous and Stochastic Sequential Decision Making Problems, Courtoux

    :param alpha: (float) the number of children of a decision node are always greater that v**alpha, where v is the number of visits to the current decision node
    :param initial_state: (DecisionNode) initial state of the tree
    :param K: exploration parameter of UCB
    :param generative_model: (fun: RandomNode -> [DecisionNode,float]) Function that gets a RandomNode and returns a DecisionNode and a float.
    :param rollout_policy: (fun: DecisionNode -> float) Function that selects an action from a DecisionNode during the rollout procedure
    :param action_sampler: (fun: DecisionNode -> float) Function that selects an action from a DecisionNode during the sampling procedure
    """

    def __init__(self,
                 alpha,
                 initial_state,
                 K,
                 generative_model,
                 rollout_policy,
                 action_sampler):

        super(SPW, self).__init__(initial_state, K, generative_model, rollout_policy, action_sampler)

        self.alpha = alpha

    def select(self, x):

        """
        Selects the action to play from the current decision node. The number of children of a DecisionNode is kept finite at all times and monotonic to the number of visits of the DecisionNode.

        :param x: (DecisionNode) current decision node
        :return: (float) action to play
        """
        if x.visits**self.alpha >= len(x.children):
            a = self.action_sampler(x)

        else:
            Q = [node.cumulative_reward/node.visits + self.K*np.sqrt(np.log(x.visits)/node.visits) if node.visits > 0 else np.inf for node in x.children.values()]

            index_best_action = np.argmax(Q)

            a = list(x.children.values())[index_best_action].action
        return a


class DPW(SPW):
    """
    Double Progressive Widening trees based on Monte Carlo Tree Search for Continuous and Stochastic Sequential Decision Making Problems, Courtoux

    :param alpha: (float) the number of children of a decision node are always greater that v**alpha, where v is the number of visits to the current decision node
    :param beta: (float) the number of outcomes of a random node is grater that v**beta where v is the number of visits of the random node
    :param initial_state: (DecisionNode) initial state of the tree
    :param K: exploration parameter of UCB
    :param generative_model: (fun: RandomNode -> [DecisionNode,float]) Function that gets a RandomNode and returns a DecisionNode and a float.
    :param rollout_policy: (fun: DecisionNode -> float) Function that selects an action from a DecisionNode during the rollout procedure
    :param action_sampler: (fun: DecisionNode -> float) Function that selects an action from a DecisionNode during the sampling procedure
    """
    def __init__(self,
                 alpha,
                 beta,
                 initial_state,
                 K,
                 generative_model,
                 rollout_policy,
                 action_sampler):

        super(DPW, self).__init__(alpha, initial_state, K, generative_model, rollout_policy, action_sampler)

        self.beta = beta

    def select_outcome(self, random_node):
        """
        The number of outcomes of a RandomNode is kept fixed at all times and increasing to the number of visits of the random_node

        :param: random_node: (RandomNode) random node from which to select the next state
        :return: (DecisionNode, float) return the next decision node and reward
        """

        if random_node.visits**self.beta >= len(random_node.children):
            (new_state, r) = self.generative_model(random_node)

            return (new_state, r)
        else:
            unnorm_probs = [child.visits for child in random_node.children.values()]
            probs = np.array(unnorm_probs)/np.sum(unnorm_probs)

            chosen_state = np.random.choice(list(random_node.children.values()), p=probs)
            return (chosen_state, chosen_state.reward)

if __name__ == "__main__":

    """
    Example to reproduce the toy case in the paper
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    DISCRETE = False

    np.random.seed(2)

    R = 0.01
    l = 1
    w = 0.7
    a = 0.7
    h = 1
    n = 10

    def reward(x):
        """
        Gets the reward observed in the decision node x

        :param x: (DecisionNode) decision node in which the agent collects the reard
        :return: (float) reward
        """
        if x.state[0] < l:
            return a
        if x.state[0] > l+w:
            return h
        return 0

    def transition(x, a):
        """
        Takes a decision node and an action and outputs another
        decision node based on the transition dynamics

        :param: x: (DecisionNode) decision node from which to compute the transition
        :param: a: (float) action taken in the decision node x
        """

        noise = np.random.uniform()
        x_new = Decision_Node(
            state=(x.state[0]+a+R*noise, x.state[1]+1),
            father=None,
            is_root=False,
            is_final=(x.state[1]+1 == 2))

        return x_new

    def generative_model(random_node):
        """
        Given a random node (state and action taken) computes the transition and gets the reward observed in the next state.

        :param random_node: (RandomNode) random node (defined by state and action) from which to generate next state
        :return: (DecisionNode, float) returns the next decisionn node and the reward observed in the next state
        """

        state_new = transition(random_node.father, random_node.action)
        r = reward(state_new)
        return (state_new, r)

    def rollout_policy(x):
        """
        Function that maps decision nodes to acions. To use while evaluationg the node value.

        :param x: (DecisionNode) From which to take the actiion
        :return: (float) the action to take at the decision node x
        """

        if DISCRETE:
            action = np.random.choice(n+1)/n
        else:
            action = np.random.uniform()
        return action

    def action_sampler(x):
        """
        Function that maps decision nodes to acions. To use while tarversing the tree.

        :param x: (DecisionNode) From which to take the actiion
        :return: (float) the action to take at the decision node x
        """

        if DISCRETE:
            action = np.random.choice(n+1)/n
        else:
            action = np.random.uniform()
        return action

    def Test_algo(Nsim, algo):
        x0 = Decision_Node((0, 0), father=None, is_root=True)

        Reward = 0
        x = x0

        while not x.is_final:

            if algo == "DPW":
                tree = DPW(alpha=0.2, beta=0.2, initial_state=x, K=2, generative_model=generative_model, rollout_policy=rollout_policy, action_sampler=action_sampler)

            else:
                tree = SPW(alpha=0.2, initial_state=x, K=2, generative_model=generative_model, rollout_policy=rollout_policy, action_sampler=action_sampler)

            action = tree.run(Nsim)

            x = transition(x, action)
            x.is_root = True

            r = reward(x)
            Reward += r

        return Reward

    Nsims = [1000, 5000, 10000, 100000]
    R_SPW = []
    R_DPW = []

    internal_sim = 3

    for N in Nsims:
        print("N: ", N)
        spw_tmp = []
        dpw_tmp = []
        for _ in tqdm(range(internal_sim)):
            spw_tmp.append(Test_algo(N, "SPW"))
            dpw_tmp.append(Test_algo(N, "DPW"))
        print("SPW")
        print(spw_tmp)

        print("DPW")
        print(dpw_tmp)

        R_SPW.append(np.mean(spw_tmp))
        R_DPW.append(np.mean(dpw_tmp))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(Nsims, R_SPW, marker="^")
    ax.plot(Nsims, R_DPW, marker="*")
    ax.set_xscale('log')
    ax.legend(["SPW", "DPW"])
    plt.show()
