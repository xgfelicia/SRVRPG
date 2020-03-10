
import argparse
import numpy as np
import theano
import theano.tensor as TT
from lasagne.updates import adam
import matplotlib.pyplot as plt

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import normalize
from rllab.sampler import parallel_sampler


from tensorboardX import SummaryWriter


#####################################################################

parser = argparse.ArgumentParser(description = "PyTorch")
parser.add_argument("--no-cuda", action = 'store_true', default = False,
						help = 'disables cuda training')
parser.add_argument("--gamma", type = float, default = 0.999)
parser.add_argument("--fg-size", type = int, default = 25)
parser.add_argument("--subiter-size", type = int, default = 3)
parser.add_argument("--batch-size", type = int, default = 10)
parser.add_argument("--max-iterations", type = int, default = 2500)
parser.add_argument("--learning-rate", type = float, default = 0.005)
parser.add_argument("--epsilon", type = float, default = 0.5)
pathsave = ""
ARGS = parser.parse_args()
print(ARGS)


def filenameGenerateSVRG(base, gamma, fg, batch, itr, m, lr):
	args = {"--gamma": gamma,
			"--fg-size": fg,
			"--batch-size": batch,
			"--max-iterations": itr,
			"--subiter-size": m,
			"--learning-rate": lr,
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


def averageSubGradient(sub_observations, sub_actions, sub_d_rewards, s_g, func_importance_weights, func_train, n_paths):
	iw = func_importance_weights(sub_observations[0], sub_actions[0])

	g = func_train(sub_observations[0], sub_actions[0], sub_d_rewards[0],
						s_g[0],s_g[1],s_g[2],s_g[3], iw)

	for ob,ac,rw in zip(sub_observations[1:],sub_actions[1:],sub_d_rewards[1:]):
		iw = func_importance_weights(ob,ac)
		g = [sum(x) for x in zip(g, func_train(ob,ac,rw,s_g[0],s_g[1],s_g[2],s_g[3],iw))]

	g = [x/n_paths for x in g]
	return g

#############################################################3


if __name__ == '__main__':
	save_policy = False

	# HYPERPARAMETERS
	# We will collect 100 trajectories per iteration
	N = ARGS.fg_size
	# Each trajectory will have at most 100 time steps
	T = 100
	# We will collect M secondary trajectories
	M = ARGS.batch_size
	# Number of sub-iterations
	m_itr = ARGS.subiter_size
	# Number of iterations
	s_tot = ARGS.max_iterations
	n_itr =  np.int(s_tot/(m_itr * M + N))

	# Set the discount factor for the problem
	discount = ARGS.gamma
	# Learning rate for the gradient update
	learning_rate =  ARGS.learning_rate
	# Number of experiments
	experiments = 10


	# normalize() makes sure that the actions for the environment lies
	# within the range [-1, 1] (only works for environments with continuous actions)
	env = normalize(CartpoleEnv())

	# Initialize a neural network policy with a single hidden layer of 8 hidden units
	policy = GaussianMLPPolicy(env.spec, hidden_sizes=(64,),learn_std=False)
	snap_policy = GaussianMLPPolicy(env.spec, hidden_sizes=(64,),learn_std=False)



	# policy.distribution returns a distribution object under rllab.distributions. It contains many utilities for computing
	# distribution-related quantities, given the computed dist_info_vars. Below we use dist.log_likelihood_sym to compute
	# the symbolic log-likelihood. For this example, the corresponding distribution is an instance of the class
	# rllab.distributions.DiagonalGaussian
	dist = policy.distribution
	snap_dist = snap_policy.distribution

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
	importance_weights_var = TT.vector('importance_weight')

	# policy.dist_info_sym returns a dictionary, whose values are symbolic expressions for quantities related to the
	# distribution of the actions. For a Gaussian policy, it contains the mean and (log) standard deviation.
	dist_info_vars = policy.dist_info_sym(observations_var)
	snap_dist_info_vars = snap_policy.dist_info_sym(observations_var)

	surr = TT.sum(- dist.log_likelihood_sym_1traj_GPOMDP(actions_var, dist_info_vars) * d_rewards_var)

	params = policy.get_params(trainable=True)
	snap_params = snap_policy.get_params(trainable=True)
	# save initial parameters
	policy_parameters = policy.get_param_values(trainable=True)

	importance_weights = dist.likelihood_ratio_sym_1traj_GPOMDP(actions_var, dist_info_vars, snap_dist_info_vars)
	grad = theano.grad(surr, params)

	eval_grad1 = TT.matrix('eval_grad0',dtype=grad[0].dtype)
	eval_grad2 = TT.vector('eval_grad1',dtype=grad[1].dtype)
	eval_grad3 = TT.col('eval_grad3',dtype=grad[2].dtype)
	eval_grad4 = TT.vector('eval_grad4',dtype=grad[3].dtype)

	surr_on1 = TT.sum(- dist.log_likelihood_sym_1traj_GPOMDP(actions_var, dist_info_vars) * d_rewards_var)
	#surr_on2 = TT.sum(- snap_dist.log_likelihood_sym_1traj_GPOMDP(actions_var, snap_dist_info_vars) * d_rewards_var )
	#grad_SVRG =[sum(x) for x in zip([eval_grad1, eval_grad2, eval_grad3, eval_grad4],
	#									theano.grad(surr_on1, params),
	#									[-1.0 * TT.mean(importance_weights_var) * x for x in theano.grad(surr_on2, snap_params)] )]

	surr_on2 = TT.sum(- snap_dist.log_likelihood_sym_1traj_GPOMDP(actions_var, snap_dist_info_vars) * d_rewards_var * importance_weights_var)
	grad_SVRG =[sum(x) for x in zip([eval_grad1, eval_grad2, eval_grad3, eval_grad4],
										theano.grad(surr_on1, params),
										[ -1.0 * x for x in theano.grad(surr_on2, snap_params)] )]


	f_train = theano.function(
		inputs = [observations_var, actions_var, d_rewards_var],
		outputs = grad
	)
	f_update = theano.function(
		inputs = [eval_grad1, eval_grad2, eval_grad3, eval_grad4],
		outputs = None,
		updates = adam([eval_grad1, eval_grad2, eval_grad3, eval_grad4], params,
						learning_rate=learning_rate, beta2 = 0.99)
	)
	f_importance_weights = theano.function(
		inputs = [observations_var, actions_var],
		outputs = importance_weights
	)

	f_update_SVRG = theano.function(
		inputs = [eval_grad1, eval_grad2, eval_grad3, eval_grad4],
		outputs = None,
		updates = adam([eval_grad1, eval_grad2, eval_grad3, eval_grad4], params,
						learning_rate=learning_rate/2, beta2 = 0.90)
	)
	f_train_SVRG = theano.function(
		inputs=[observations_var, actions_var, d_rewards_var, eval_grad1, eval_grad2, eval_grad3, eval_grad4,importance_weights_var],
		outputs=grad_SVRG,
	)



	all_data = []
	for k in range(experiments):
		snap_policy.set_param_values(policy_parameters, trainable = True)
		policy.set_param_values(policy_parameters, trainable = True)

		if save_policy:
			np.savetxt("policy_novar.txt",snap_policy.get_param_values(trainable=True))


		avg_return = np.zeros(n_itr*(m_itr+1))
		for j in range(n_itr):
			parallel_sampler.populate_task(env, snap_policy)
			paths = parallel_sampler.sample_paths_on_trajectories(snap_policy.get_param_values(),N,T,show_bar=False)
			# theta0 = snap_policy.get_param_values(trainable = True)

			# pull results from paths into lists of trajectory results
			observations = [p["observations"] for p in paths]
			actions = [p["actions"] for p in paths]
			rewards = [p["rewards"] for p in paths]


			d_rewards = discountRewards(rewards, discount)
			s_g = averageGradient(observations, actions, d_rewards, f_train, len(paths))
			f_update(s_g[0],s_g[1],s_g[2],s_g[3])

			avg_return[j*(m_itr+1)] = np.mean([sum(p["rewards"]) for p in paths])

			for i in range(m_itr):
				parallel_sampler.populate_task(env, policy)
				sub_paths = parallel_sampler.sample_paths_on_trajectories(policy.get_param_values(),M,T,show_bar=False)

				sub_observations=[p["observations"] for p in sub_paths]
				sub_actions = [p["actions"] for p in sub_paths]
				sub_rewards = [p["rewards"] for p in sub_paths]

				sub_d_rewards = discountRewards(sub_rewards, discount)

				g = averageSubGradient(sub_observations, sub_actions, sub_d_rewards, s_g,
											f_importance_weights, f_train_SVRG, len(sub_paths))

				f_update(g[0], g[1], g[2], g[3])
				avg_return[j*(m_itr+1)+i+1] = np.mean([sum(p["rewards"]) for p in sub_paths])


			snap_policy.set_param_values(policy.get_param_values(trainable=True), trainable=True)
			print(str(j) +' Average Return:', avg_return[j*(m_itr+1)])

		all_data.append(avg_return)


	all_data_mean = [np.mean(x) for x in zip(*all_data)]

	name = filenameGenerateSVRG("svrg-beta9099-beta9090-64hn-div2", discount, N, M, s_tot, m_itr, learning_rate)
	writer = SummaryWriter()
	for i in range(len(all_data_mean)):
		writer.add_scalar(tag = name, scalar_value = all_data_mean[i], global_step = i)
