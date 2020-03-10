
import gym
import numpy as np
import collections
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions
from torch.distributions import Categorical


from rllab_wrapper import *


# edited from rllab
def set_all_seeds(seed):
	seed %= 4294967294
	global seed_
	seed_ = seed

	import torch
	torch.manual_seed(0)
	np.random.seed(seed)

	try:
		import tensorflow as tf
		tf.set_random_seed(seed)
	except Exception as e:
		print(e)

	try:
		import lasagne
		random.seed(seed)
		lasagne.random.set_rng(np.random.RandomState(seed))

	except Exception as e:
		print(e)

	print( 'using seed %s' % (str(seed)) )


# policy neural net class
class Policy(nn.Module):
	def __init__(self, state_size, action_size):
		super(Policy, self).__init__()
		hidden = 8


		self.run = nn.Sequential(
			nn.Linear(state_size, hidden),
			nn.Tanh(),
			nn.Linear(hidden, hidden),
			nn.Tanh(),
			nn.Linear(hidden, action_size),
			# nn.Tanh(),
		)


	def forward(self, x):
		return self.run(x)


#############################################################

# agent class
class Agent():
	def __init__(self, state_size, action_size, N, B, m):
		self.model = Policy(state_size, action_size)
		self.learning_rate_eta = 0.01
		self.learning_rate_tau = 0.005
		self.gamma = 0.99

		self.N = N
		self.B = B
		self.m = m
		self.freeze = False


		# Lists to store hyper-params
		self.params = list(self.model.parameters())

		self.eta = []
		self.tau = []

		self.back_eta = []
		self.back_tau = []
		self.inner_eta = []
		self.inner_tau = []

		self.reward_history = []

		self.state_size = state_size
		self.action_size = action_size

		# Initialize parameters
		for p in self.params:
			self.eta.append(torch.zeros(p.size()) + 0)
			self.tau.append(torch.ones(p.size()) * 2)
			p.data = torch.normal(self.eta[-1], self.tau[-1])



	def reset_outer_traj_theta(self):
		self.thetas = []
		self.trajs = []

	def reset_inner_traj_theta(self):
		self.inner_thetas = []
		self.inner_trajs = []

	def reset_outer_grad(self):
		self.eta_grad = []
		self.tau_grad = []

	def reset_inner_grad(self):
		self.inner_eta_grad = []
		self.inner_tau_grad = []


	def predict(self, state):
		action = self.model.forward(state)

		# action = action.argmax()
		# print(action)
		return action.data



	def update_policy_out(self, max_reward):
		temp_reward = []

		# initialize gradients
		for p in self.params:
			self.eta_grad.append(torch.zeros(p.size()))
			self.tau_grad.append(torch.zeros(p.size()))

		# calculate gradient for each individual trajectory before doing update
		for j in range(self.N):
			states = []
			actions = []
			raw_rewards = []

			# extract trajectory info
			for val in self.trajs[j]:
				states.append(val.state)
				actions.append(val.action)
				raw_rewards.append(val.reward)


			# Discount future rewards with gamma
			t = 1.0  # start with discount = 1.0
			rewards = []
			for y in raw_rewards:
				rewards.append(y * t)
				t *= self.gamma

			rewards = torch.FloatTensor([np.sum(rewards)])

			self.reward_history.append(rewards)
			temp_reward.append(np.sum(raw_rewards))

			if (np.sum(raw_rewards) >= max_reward):
				print("Max reward: ", np.sum(raw_rewards))
				self.freeze = True
				for k in range(len(self.params)):
					 self.params[k].data = self.thetas[j][k]

				REWARDS.append(np.max(temp_reward))
				return
			else:
				print(np.sum(raw_rewards))
				self.freeze = False


			_r = float(preprocessing.scale(np.array(self.reward_history))[-1])

			# calculate values for gradients
			for i in range(len(self.params)):
				log_eta = (self.params[i].data - self.eta[i])
				log_tau = ((self.params[i].data - self.eta[i])**2 - self.tau[i]**2) / self.tau[i]

				self.eta_grad[i] += log_eta * _r
				self.tau_grad[i] += log_tau * _r

		self.back_eta = self.eta
		self.back_tau = self.tau

		for i in range(len(self.params)):
			self.eta[i] = self.eta[i] + ( self.learning_rate_eta * torch.FloatTensor([1.0 / self.N]) * self.eta_grad[i])
			self.tau[i] = self.tau[i] + ( self.learning_rate_tau * torch.FloatTensor([1.0 / self.N]) * self.tau_grad[i])

		REWARDS.append(np.mean(temp_reward))
		self.reset_outer_traj_theta()
		self.reset_outer_grad()



	def update_policy_inner(self, max_reward):
		temp_reward = []

		for p in self.params:
			self.inner_eta_grad.append(torch.zeros(p.size()))
			self.inner_tau_grad.append(torch.zeros(p.size()))

		# calculate gradient for each individual trajectory before doing update
		for j in range(self.B):
			states = []
			actions = []
			raw_rewards = []

			# extract trajectory info
			for val in self.inner_trajs[j]:
				states.append(val.state)
				actions.append(val.action)
				raw_rewards.append(val.reward)


			# Discount future rewards with gamma
			t = 1.0  # start with discount = 1.0
			rewards = []
			for y in raw_rewards:
				rewards.append(y * t)
				t *= self.gamma

			rewards = torch.FloatTensor([np.sum(rewards)])

			self.reward_history.append(rewards)
			temp_reward.append(np.sum(raw_rewards))

			if (np.sum(raw_rewards) >= max_reward ):
				print("  Raw reward: ", np.sum(raw_rewards))
				self.freeze = True
				for k in range(len(self.params)):
					 self.params[k].data = self.inner_thetas[j][k]

				REWARDS.append(np.max(temp_reward))
				return
			else:
				print(" ", np.sum(raw_rewards))
				self.freeze = False


			_r = float(preprocessing.scale(np.array(self.reward_history))[-1])

			# calculate values for gradients
			for i in range(len(self.params)):
				log_eta_t0 = (self.params[i].data - self.eta[i])
				log_tau_t0 = ((self.params[i].data - self.eta[i])**2 - self.tau[i]**2) / self.tau[i]

				log_eta_t1 = (self.params[i].data - self.inner_eta[i])
				log_tau_t1 = ((self.params[i].data - self.inner_eta[i])**2 - self.inner_tau[i]**2) / self.inner_tau[i]

				# iw: target/behaviorial
				weightn = (1.0/self.tau[i]) * torch.exp( -(self.params[i].data - self.eta[i])**2 / (2 * self.tau[i]**2))
				weightd = (1.0/self.inner_tau[i]) * torch.exp(-(self.params[i].data - self.inner_eta[i])**2 / (2 * self.inner_tau[i]**2))

				weight = weightn / weightd

				self.inner_eta_grad[i] += (log_eta_t1 * _r) - (weight) * (log_eta_t0 * _r)
				self.inner_tau_grad[i] += (log_tau_t1 * _r) - (weight) * (log_tau_t0 * _r)

		# iterate forwards
		self.eta = self.inner_eta
		self.tau = self.inner_tau

		for i in range(len(self.params)):
			self.inner_eta[i] = self.inner_eta[i] + ( self.learning_rate_eta * torch.FloatTensor([1.0 / self.B]) * self.inner_eta_grad[i])
			self.inner_tau[i] = self.inner_tau[i] + ( self.learning_rate_tau * torch.FloatTensor([1.0 / self.B]) * self.inner_tau_grad[i])

		REWARDS.append(np.mean(temp_reward))
		self.reset_inner_traj_theta()
		self.reset_inner_grad()



	def train(self, episodes, horizon, max_reward):
		# parameters and trajectories
		self.reset_outer_traj_theta()
		self.reset_inner_traj_theta()
		self.reset_outer_grad()
		self.reset_inner_grad()

		for episode in range(episodes):
			print("Episode: ", episode)
			#print(self.eta[0], self.tau[0])
			#print("length: ", len(self.eta_grad), len(self.inner_eta_grad))

			# sample N policy parameters
			for i in range(self.N):
				temp = []

				for mean, var, p in zip(self.eta, self.tau, self.params):
					if not self.freeze:
						temp.append( torch.normal(mean, var) )
					else:
						temp.append( p.data )

				self.thetas.append(temp)
			# print(self.thetas)

			# sample one trajectory from each policy (N total)
			for theta in self.thetas:
				for i in range(len(self.params)):
					self.params[i].data = theta[i]

				state = env.reset()
				traj = []

				for time in range(horizon):
					# get action using NN to predict
					state = torch.FloatTensor(state)
					action = self.predict(state)

					# Step through environment using chosen action
					next_state, reward, done, _ = env.step(np.array(action))

					# save results
					# if episode done, start new env
					data = Transition(state, action, reward)
					traj.append(data)
					state = next_state

					if done:
						break

				self.trajs.append(traj)

			# update policy after N trajectories done
			self.update_policy_out(max_reward)


			################################################
			# Inner subiterations

			# update gradients from outer loop
			if self.m > 0:
				self.inner_eta = self.eta
				self.inner_tau = self.tau
				# self.eta = self.back_eta
				# self.tau = self.back_tau


			for i_itr in range(self.m):
				print( "  Subiteration: ", i_itr)
				#print(self.eta[0], self.tau[0])
				#print("Inner: ", self.inner_eta[0], self.inner_tau[0])

				# sample B policy parameters
				for j in range(self.B):
					temp = []

					for mean, var, p in zip(self.inner_eta, self.inner_tau, self.params):
						if not self.freeze:
							temp.append( torch.normal(mean, var) )
						else:
							temp.append( p.data )

					self.inner_thetas.append(temp)

				# sample one trajectory from each policy (B total)
				for theta in self.inner_thetas:
					for i in range(len(self.params)):
						self.params[i].data = theta[i]

					state = env.reset()
					traj = []

					for time in range(horizon):
						# get action using NN to predict
						state = torch.FloatTensor(state)
						action = self.predict(state)

						# Step through environment using chosen action
						next_state, reward, done, _ = env.step(np.array(action))
						# next_state = torch.FloatTensor([next_state])
						# reward = torch.FloatTensor([reward])

						# save results
						# if episode done, start new env
						data = Transition(state, action, reward)
						traj.append(data)
						state = next_state

						if done:
							break

					self.inner_trajs.append(traj)

				# update policy after B trajectories done
				self.update_policy_inner(max_reward)



##############################################################



if __name__ == '__main__':
	Transition = collections.namedtuple('Transition',
							('state', 'action', 'reward'))

	experiments = 5
	ALL_REWARDS = []
	for i in range(experiments):
		REWARDS = []

		env = gym.make('Pendulum-v0')

		#set_all_seeds(0)

		# N = batch size, B = mini batch size, m = sub iteration
		agent = Agent(env.observation_space.shape[0], env.action_space.shape[0], N = 50, B = 10, m = 2)

		agent.train(episodes = 50, horizon = 200, max_reward = -200)

		ALL_REWARDS.append(REWARDS)

	ALL_REWARDS = np.mean(np.array(ALL_REWARDS), axis = 0)
	np.savetxt("pendulum-spider-policy-3500t.csv", np.transpose( np.array(ALL_REWARDS) ), delimiter = ',')
