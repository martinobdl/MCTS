import numpy as np
import pdb
from tqdm import tqdm_notebook as tqdm

class Decision_Node:
    def __init__(self,
                state,
                father = None,
                is_root = False,
                is_final = False):
        """
        state: tuple, defining the state
        father: RandomNode
        is_root: bool
        is_final: bool
        """
        self.state = state
        self.children = {}
        self.is_final = is_final
        self.visits = 0
        self.father = father
        self.is_root = is_root

    def add_children(self, random_node):
        """
        adds a RandomNode  object to the dictionary of childrens (key is the action)
        """
        self.children[str(random_node.action)] = random_node

    def reward(self, r):
        """
        """
        self.reward = r

    def __repr__(self):
        s = ""
        for item in self.__dict__.items():
            if item[0] == "children":
                s+=" "+"(children, "+str(len(self.children))+")"
            elif item[0] == "father":
                pass
            else:
                s+=" "+str(item)
        return s

class Random_Node:
    def __init__(self,
                state,
                action,
                father=None):
        """
        state: tuple, defining the state
        action: action taken in the decision node
        father: DecisionNode
        is_root: bool
        is_final: bool
        """
        self.state = state
        self.action = action
        self.children = {}
        self.cumulative_reward = 0
        self.visits = 0
        self.father = father

    def reward(self, reward):
        self.reward = reward

    def add_children(self, decision_Node):
        """
        adds a DecisionNode object to the dictionary of childrens (key is the state)
        """
        self.children[str(decision_Node.state)] = decision_Node

    def __repr__(self):
        s = ""
        for item in self.__dict__.items():
            if item[0] == "children":
                s+=" "+"(children, "+str(len(self.children))+")"
            elif item[0] == "father":
                pass
            else:
                s+=" "+str(item)
        return s

class MCTS:
    """
    Base class for MCTS based on Monte Carlo Tree Search for Continuous and Stochastic Sequential Decision Making Problems, Courtoux

    methods:
        grow_tree
        evaluate
        select
        select_outcome
        best_action
        run

    """
    def __init__(self,
                 initial_state,
                 K,
                 generative_model,
                 rollout_policy,
                 action_sampler):
        """
        intial_state: DecisionNode
        K: exploration parameter
        generative_model: RandomNode -> DecisionNode
        rollout_policy: DecisionNode -> action
        action_sampler: DecisionNode -> action
        """
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
                new_random_node = Random_Node(x, a, father = decision_node)
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

        decision_node.visits += 1

        while not decision_node.is_root:
            random_node = decision_node.father
            cumulative_reward += random_node.reward
            random_node.cumulative_reward += cumulative_reward
            random_node.visits += 1
            decision_node = random_node.father
            decision_node.visits += 1

    def evaluate(self, state):
        """
        evaluates a DecionNode playing until an terminal node using the rollotPolicy
        return: reward
        """
        R = 0
        while not state.is_final:
            a = self.rollout_policy(state)
            new_random_node = Random_Node(state.state, a, father=state)
            (state, r) = self.generative_model(new_random_node)
            R += r

        return R

    def select_outcome(self, random_node):
        """
        given a RandomNode returns a DecisionNode
        """
        return self.generative_model(random_node)

    def select(self, state):
        raise NotImplemented

        # Q = [node.cumulative_reward/node.visits + self.K*np.sqrt(np.log(state.visits)/node.visits)
        #      for node in state.children]

        # index_best_action = np.argmax(Q)

        # return state.children[index_best_action].action

    def best_action(self):
        """
        At the end of the simulations returns the most visited action
        """
        number_of_visits_children = [node.visits for node in self.root.children.values()]
        index_best_action = np.argmax(number_of_visits_children)

        return list(self.root.children.values())[index_best_action].action

    def run(self, Nsim, progress_bar=False):
        """
        Expand the tree and return the bet action
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
    Simple Progressive Widening
    Best suited for continuous action sapce and discrete state space

    """
    def __init__(self,
                 alpha,
                 initial_state,
                 K,
                 generative_model,
                 rollout_policy,
                 action_sampler):

        super(SPW, self).__init__(initial_state,
                                 K,
                                 generative_model,
                                 rollout_policy,
                                 action_sampler)

        self.alpha = alpha

    def select(self, decision_node):
        """
        The number of children of a DecisionNode is kept finite at all times and monotonic to the number of visits of the DecisionNode
        """
        if decision_node.visits**self.alpha >= len(decision_node.children):
            a = self.action_sampler(decision_node)

        else:
            Q = [node.cumulative_reward/node.visits + self.K*np.sqrt(np.log(decision_node.visits)/node.visits)
             if node.visits>0 else np.inf for node in decision_node.children.values()]

            index_best_action = np.argmax(Q)

            a = list(decision_node.children.values())[index_best_action].action
        return a

class DPW(SPW):
    def __init__(self,
                 alpha,
                 beta,
                 initial_state,
                 K,
                 generative_model,
                 rollout_policy,
                 action_sampler):

        super(DPW, self).__init__(alpha,
                                initial_state,
                                K,
                                generative_model,
                                rollout_policy,
                                action_sampler)

        self.beta = beta

    def select_outcome(self, random_node):
        """
        The number of outcomes of a RandomNode is kept fixed at all times and increasing to the number of visits of the random_node

        return a new_decisionNode and a reward
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

    #pdb.set_trace()

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
        x: Decision_Node obj
        return: float reward
        """
        if x.state[0]<l:
            return a
        if x.state[0]>l+w:
            return h
        return 0

    def transition(x, a):
        """
        takes a decision node and an action and outputs another
        decision node based on the transition dynamics

        x: state_object
        a: numeric
        """
        noise = np.random.uniform()
        # noise = np.random.choice(5)/5
        x_new = Decision_Node(
            state = (x.state[0]+a+R*noise, x.state[1]+1),
            father = None,
            is_root = False,
            is_final = (x.state[1]+1 == 2))

        return x_new

    def generative_model(random_node):
        """
        state: state_variable
        a: numeric
        """
        state_new = transition(random_node.father, random_node.action)
        r = reward(state_new)
        return (state_new, r)

    def rollout_policy(x):
        """
        x: numeric
        return a random legal action
        """
        if DISCRETE:
            action = np.random.choice(n+1)/n
        else:
            action = np.random.uniform()
        return action

    def action_sampler(state):
        if DISCRETE:
            action = np.random.choice(n+1)/n
        else:
            action = np.random.uniform()
        return action

    def Test_algo(Nsim, algo):
        x0 = Decision_Node((0,0), father=None, is_root = True)

        Reward = 0
        x = x0

        while not x.is_final:

            if algo=="DPW":
                tree = DPW(alpha = 0.2,
                            beta = 0.2,
                            initial_state = x,
                            K = 2,
                            generative_model    = generative_model,
                            rollout_policy      = rollout_policy,
                            action_sampler      = action_sampler)

            else:
                tree = SPW(alpha = 0.2,
                            initial_state = x,
                            K = 2,
                            generative_model    = generative_model,
                            rollout_policy      = rollout_policy,
                            action_sampler      = action_sampler)

            action = tree.run(Nsim)

            x = transition(x, action)
            x.is_root = True

            r = reward(x)
            Reward += r

        return Reward

    Nsims = [1000,5000,10000,100000,1000000,2000000]
    R_SPW=[]
    R_DPW=[]

    internal_sim = 10

    for N in Nsims:
        print("N: ",N)
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
    ax.plot(Nsims, R_SPW, marker = "^")
    ax.plot(Nsims, R_DPW, marker = "*")
    ax.set_xscale('log')
    ax.legend(["SPW","DPW"])
    plt.show()


