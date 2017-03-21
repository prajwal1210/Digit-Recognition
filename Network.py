import random
import numpy as np
class network(object):

	def __init__(self,sizes):
		self.numlayers=len(sizes)
		self.sizes=sizes
		self.biases=[np.random.randn(y,1) for y in sizes[1:]]
		self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

	def feedforward(self,x):
		a=x
		for w,b in zip(self.weights,self.biases):
			a=sigmoid(np.dot(w,a)+b)
		return a

	def train(self,training_data,iters,mini_batch_size,eta,test_data=None):
		if test_data: n_test=len(test_data)
		n=len(training_data)
		for i in xrange(iters):
			mini_batches=[training_data[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch,eta)
			if(test_data):
				print "Iteration {0}: {1}/{2}".format(i,self.evaluate(test_data),n_test)
			else:
				print "Iteration {0} complete".format(i)

	def update_mini_batch(self,mini_batch,eta):
		tdelta_w=[np.zeros(w.shape) for w in self.weights]
		tdelta_b=[np.zeros(b.shape) for b in self.biases]
		for x,y in mini_batch:
			delta_w, delta_b= self.backpropogate(x,y)
			tdelta_w=[w+dw for w,dw in zip(tdelta_w,delta_w)]
			tdelta_b=[b+db for b,db in zip(tdelta_b,delta_b)]
		self.weights=[w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights,tdelta_w)]
		self.biases=[b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases,tdelta_b)]

	def cost_derivative(self, output_activations, y):
		return (output_activations-y)

	def backpropogate(self,x,y):
		delta_w=[np.zeros(w.shape) for w in self.weights]
		delta_b=[np.zeros(b.shape) for b in self.biases]
		activation=x
		activations=[x]
		zs=[]
		for w,b in zip(self.weights,self.biases):
			z=np.dot(w,activation)+b
			zs.append(z)
			activation=sigmoid(z)
			activations.append(activation)
		delta=self.cost_derivative(activations[-1],y)*sigmoid_prime(zs[-1])
		delta_b[-1]=delta
		delta_w[-1]=np.dot(delta,activations[-2].transpose())
		for l in xrange(2, self.numlayers):
			z = zs[-l]
        	sp = sigmoid_prime(z)
        	delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
         	delta_b[-l] = delta
        	delta_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (delta_w, delta_b)

	def evaluate(self,test_data):
		test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)


def sigmoid(z):
		return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
		return sigmoid(z)*(1-sigmoid(z))
