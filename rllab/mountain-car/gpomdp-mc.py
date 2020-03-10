
import argparse
import numpy as np
import theano
import theano.tensor as TT
from lasagne.updates import adam
import matplotlib.pyplot as plt
from rllab.envs.gym_env import GymEnv
import lasagne.nonlinearities as NL


from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import normalize
from rllab.sampler import parallel_sampler
from rllab.misc import ext


from tensorboardX import SummaryWriter

#####################################################################

parser = argparse.ArgumentParser(description = "PyTorch")
parser.add_argument("--no-cuda", action = 'store_true', default = False,
                        help = 'disables cuda training')
parser.add_argument("--gamma", type = float, default = 0.999)
parser.add_argument("--fg-size", type = int, default = 10)
parser.add_argument("--learning-rate", type = float, default = 0.005)
parser.add_argument("--max-iterations", type = int, default = 2500)

parser.add_argument("--subiter-size", type = int, default = 3)
parser.add_argument("--batch-size", type = int, default = 10)
parser.add_argument("--epsilon", type = float, default = 0.5)
pathsave = ""
ARGS = parser.parse_args()
print(ARGS)


def filenameGenerateSVRG(base, gamma, fg, lr, maxitr):
    args = {"--gamma": gamma,
            "--fg-size": fg,
            "--learning-rate": lr,
            "--max-iterations": maxitr
            }

    name = base + "".join([ key + str(value) for key, value in sorted(args.items())] )
    return name


def discountRewards(rewards, discount):
    temp = list()

    for x in rewards:
        z = list()
        t = 1  # start with discount = 1

        for y in x:
            z.append(y * t)
            t *= discount

        temp.append(np.array(z))

    return temp


def averageGradient(observations, actions, d_rewards, func_train, n_paths):
    # output the list of gradients and sum them over all paths
    s_g = func_train(observations[0], actions[0], d_rewards[0])

    for ob,ac,rw in zip(observations[1:], actions[1:], d_rewards[1:]):
        s_g = [sum(x) for x in zip(s_g, func_train(ob, ac, rw)) ]

    # get the average gradient value over all paths before gradient update with ADAM
    s_g = [x/n_paths for x in s_g]

    return s_g


#######################################################

if __name__ == '__main__':
    import_policy = True
    save_policy = False

    # HYPERPARAMETERS
    # We will collect N trajectories per iteration
    N = ARGS.fg_size
    # Each trajectory will have at most T time steps
    T = 1000
    # Number of iterations per experiment
    n_itr = int(ARGS.max_iterations/N)
    # Set the discount factor for the problem
    discount = ARGS.gamma
    # Learning rate for the gradient update
    learning_rate = ARGS.learning_rate
    # Number of times to run experiment
    experiments = 10


    # normalize() makes sure that the actions for the environment lies
    # within the range [-1, 1] (only works for environments with continuous actions)
    env = normalize(GymEnv("MountainCarContinuous-v0"))

    # Initialize a neural network policy with a single hidden layer of 8 hidden units
    # do not learn the standard deviation
    policy = GaussianMLPPolicy(env.spec, hidden_sizes = (64,), learn_std=False, hidden_nonlinearity = NL.tanh)
    parallel_sampler.populate_task(env, policy)

    # policy.distribution returns a distribution object under rllab.distributions. It contains many utilities for computing
    # distribution-related quantities, given the computed dist_info_vars. Below we use dist.log_likelihood_sym to compute
    # the symbolic log-likelihood. For this example, the corresponding distribution is an instance of the class
    # rllab.distributions.DiagonalGaussian
    dist = policy.distribution


    observations_var = env.observation_space.new_tensor_variable(
        'observations',
        # It should have 1 extra dimension since we want to represent a list of observations
        extra_dims=1
    )
    actions_var = env.action_space.new_tensor_variable(
        'actions',
        extra_dims=1
    )
    d_rewards_var = TT.vector('d_rewards')


    # policy.dist_info_sym returns a dictionary, whose values are symbolic expressions for quantities related to the
    # distribution of the actions.
    # For a Gaussian policy, it contains the mean and (log) standard deviation.
    dist_info_vars = policy.dist_info_sym(observations_var)

    # negate the objective for minimization problem
    surr = TT.sum(-dist.log_likelihood_sym_1traj_GPOMDP(actions_var, dist_info_vars) * d_rewards_var)
    # get the list of trainable parameters
    params = policy.get_params(trainable=True)
    # save initial parameters
    policy_parameters = policy.get_param_values(trainable=True)
    grad = theano.grad(surr, params)

    eval_grad1 = TT.matrix('eval_grad0',dtype=grad[0].dtype)    # (4, 8) hiddenlayer.w = LI.GlorotUniform() aka Xavier Uniform Init
    eval_grad2 = TT.vector('eval_grad1',dtype=grad[1].dtype)    # (8, )  hiddenlayer.b = LI.Constant(0.),
    eval_grad3 = TT.col('eval_grad3',dtype=grad[2].dtype)       # (8, 1) output.w = LI.GlorotUniform(),
    eval_grad4 = TT.vector('eval_grad4',dtype=grad[3].dtype)    # (1, )  output.b = LI.Constant(0.),

    f_train = theano.function(
        inputs = [observations_var, actions_var, d_rewards_var],
        outputs = grad
    )
    f_update = theano.function(
        inputs = [eval_grad1, eval_grad2, eval_grad3, eval_grad4],
        outputs = None,
        updates = adam([eval_grad1, eval_grad2, eval_grad3, eval_grad4], params,
            learning_rate=learning_rate, beta1 = 0.90, beta2 = 0.999)
    )



    all_data = []
    for i in range(experiments):
        if import_policy:
            print("Importing policy...")
            policy.set_param_values(np.loadtxt('policy-mc-64hn-5-7.txt'), trainable=True)
        else:
            policy.set_param_values(policy_parameters, trainable = True)

        if save_policy:
            np.savetxt("policy-mc-64hn.txt", policy.get_param_values(trainable=True))

        avg_return = np.zeros(n_itr)
        for j in range(n_itr):
            # sample paths using policy parameters
            paths = parallel_sampler.sample_paths_on_trajectories(policy.get_param_values(), N, T, show_bar=False)

            # pull results from paths into lists of trajectory results
            # eg: observations is a list with length N (for each trajectory)
            observations = [p["observations"] for p in paths]
            actions = [p["actions"] for p in paths]
            rewards = [p["rewards"] for p in paths]

            d_rewards = discountRewards(rewards, discount)

            # output the list of gradients and sum them over all paths
            s_g = averageGradient(observations, actions, d_rewards, f_train, len(paths))
            f_update(s_g[0],s_g[1],s_g[2],s_g[3])

            avg_return[j] = np.mean([sum(p["rewards"]) for p in paths])
            if (j % 10 == 0):
                print(str(j) +' Average Return:', avg_return[j])

        all_data.append(avg_return)


    all_data_mean = [np.mean(x) for x in zip(*all_data)]

    name = filenameGenerateSVRG("gpomdp-beta90999-64hn", discount, N, learning_rate, ARGS.max_iterations)
    writer = SummaryWriter()
    for i in range(len(all_data_mean)):
        writer.add_scalar(tag = name, scalar_value = all_data_mean[i], global_step = i)
