import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf
import os
%matplotlib inline

# Design Game Environment:
class gameOb():
	def _init_(self, coordinates, size, intensity, channel, reward, name):
		self.x = coordinates[0]
		self.y = coordinates[1]
		self.size = size
		self.intensity = intensity
		self.channel = channel
		self.reward = reward
		self.name = name

class gameEnv():
	def _init_(self, size):
		self.sizeX = size
		self.sizeY = size
		self.actions = 4
		self.objects = []
		a = self.reset()
		plt.imshow(a, interpolation = "nearest")

	def reset(self):
		self.objects = []
		hero = gameOb(self.newPosition(), 1, 1, 2, None, 'hero')
		self.objects.append(hero)
		goal = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
		self.objects.append(goal)
		hole = gameOb(self.newPosition(), 1, 1, 0, -1, 'fire')
		self.objects.append(hole)
		goal2 = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
		self.objects.append(goal2)
		hole2 = gameOb(self.newPosition(), 1, 1, 0, -1, 'fire')
		self.objects.append(hole2)
		goal3 = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
		self.objects.append(goal3)
		goal4 = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
		self.objects.append(goal4)
		
		state = self.renderEnv()
		self.state = state
		
		return state

	def moveChar(self, direction):
		hero = self.objects[0]
		heroX = hero.x
		heroY = hero.y
		if direction == 0 and hero.y >= 1:
			hero.y -= 1
		if direction == 1 and hero.y <= self.sizeY - 2:
			hero.y += 1
		if direction = 2 and hero.x >= 1:
			hero.x -= 1
		if direction = 3 and hero.x <= self.sizeX - 2:
			hero.x += 1
		self.objects[0] = hero

	def newPosition(self):
		iterables = [range(self.sizeX), range(self.sizeY)]
		points = []
		for t in itertools.product(*iterables):
			points.append(t)
		currentPositions = []
		for objectA in self.objects:
			if (objectA.x, objectA.y) not in currentPositions:
				currentPositions.append((objectA.x, objectA.y))
		for pos in currentPositions:
			points.remove(pos)
		location = np.random.choice(range(len(points)), replace = False)
		return points[location]

	def checkGoal(self):
		others = []
		for obj in self.objects:
			if obj.name == 'hero':
				hero = obj
			else:
				others.append(obj)

		for other in others:
			if hero.x == other.x and hero.y == other.y:
				self.objects.remove(other)
				if other.reward == 1:
					self.objects.append(gameOb(self.newPosition(), 1, 1, 1, 1, 'goal'))
				else:
					self.objects.append(gameOb(self.newPosition(), 1, 1, 0, -1, 'fire'))
				return other.reward, else

		return 0.0, False

	def renderEnv(self):
		a = np.ones([self.sizeY+2, self.sizeX+2, 3])
		a[1: -1, 1: -1, :] = 0
		hero = None
		for item in self.objects:
			a[item.y+1:item.y+item.size+1, item.x+1: item.x+item.size+1, item.channel] = item.intensity
			b = scipy.misc.imresize(a[:,:,0], [84,84,1], interp = 'nearest')
			c = scipy.misc.imresize(a[:,:,1], [84,84,1], interp = 'nearest')
			d = scipy.misc.imresize(a[:,:,2], [84,84,1], interp = 'nearest')
			a = np.stack([b,c,d], axis = 2)
			return a

	def step(self, action):
		self.moveChar(action)
		reward, done = self.checkGoal()
		state = self.renderEnv()
		return state, reward, done

env = gameEnv(size = 5)

# Design DQN:
class QNetwork():
	def _init_(self, h_size):
		self.scalarInput = tf.placeholder(shape = [None, 21168], dtype = tf.float32)
		self.imageIn = tf.reshape(self.scalarInput, shape = [-1, 84, 84, 3])
		self.conv1 = tf.contrib.layers.convolution2d(inputs = self.imageIn, num_outputs = 32, kernel_size = [8, 8], stride = [4, 4], padding = 'VALID', biases_initializer = None)
		self.conv2 = tf.contrib.layers.convolution2d(inputs = self.conv1, num_outputs = 64, kernel_size = [4, 4], stride = [2, 2], padding = 'VALID', biases_initializer = None)
		self.conv3 = tf.contrib.layers.convolution2d(inputs = self.conv2, num_outputs = 64, kernel_size = [3, 3], stride = [1, 1], padding = 'VALID', biases_initializer = None)
		self.conv4 = tf.contrib.layers.convolution2d(inputs = self.conv3, num_outputs = 32, kernel_size = [7, 7], stride = [1, 1], padding = 'VALID', biases_initializer = None)

		self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
		self.streamA = tf.contrib.layers.flatten(self.streamAC)
		self.streamV = tf.contrib.layers.flatten(self.streamVC)
		self.AW = tf.Variable(tf.random_normal([h_size//2, env.actions]))
		self.VW = tf.Variable(tf.random_normal([h_size//2, 1]))
		self.Advantage = tf.matmul(self.streamA, self.AW)
		self.Value = tf.matmul(self.streamV, self.VW)

		self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, reduction_indices = 1, keep_dims = True))
		self.predict = tf.argmax(self.Qout, 1)		

