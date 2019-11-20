import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class RBM():

	def __init__(self, learningRate, k, visibleSize, hiddenSize):
		self.k = k	# number of gibbs sampling execution
		self.learningRate = learningRate
		self.visibleSize = visibleSize
		self.hiddenSize = hiddenSize
		self.visibleBias = tf.Variable(tf.zeros([1, self.visibleSize]),tf.float32, name="visible_bias")
		self.hiddenBias = tf.Variable(tf.zeros([1, self.hiddenSize]),tf.float32, name="hidden_bias")
		self.weight = tf.Variable(tf.random_normal([self.visibleSize, self.hiddenSize], mean=0., stddev=4 * np.sqrt(6. / (self.visibleSize + self.hiddenSize))), name="weights")

	def gibbsSampling(self,negativeVisble):
		for k in range(self.k):
			negativeHidden = np.random.binomial(n=1, p=self.pInputGivenHidden(negativeVisble))
			negativeVisble = np.random.binomial(n=1, p=self.pHiddenGivenInput(negativeHidden))
		return negativeVisble

	def sigm(self,x):
		return 1/(1 + tf.exp(-x))

	def pInputGivenHidden(self,input):
		return self.sigm(tf.add(self.visibleBias,tf.matmul(self.weight,input)))

	def pHiddenGivenInput(self,hidden):
		return self.sigm(tf.add(self.hiddenBias,tf.matmul(hidden,self.weight)))

	def updateWeight(self, negativeVisible, negativeHidden):
		self.visibleBias += self.learningRate*(self.visibleLayer - negativeVisible)
		self.hiddenBias += self.learningRate*(self.hiddenLayer - negativeHidden)
		self.weight += tf.matmul(tf.transpose(self.visibleLayer), self.hiddenLayer) - np.outer(negativeVisible,negativeHidden)


	def training(self,input):
		self.visibleLayer = tf.reshape(input,[1,784])
		self.hiddenLayer = self.pHiddenGivenInput(self.visibleLayer)

		visible = np.copy(self.visibleLayer)
		negativeVisible = self.gibbsSampling(visible)
		negativeHidden = self.pHiddenGivenInput(negativeVisible)
		self.updateWeight(negativeVisible,negativeHidden)
		return self.weight

	def plot(self):
		return 0

	def saveModel(self):
		save = tf.train.Saver({"weights":self.w, "visible_bias":self.visibleBias, "hidden_bias":self.hiddenBias})
		return save

if __name__ == '__main__':
	numSteps = 1
	(visibles, classifications), (Testvisibles, testClassifications) = tf.keras.datasets.mnist.load_data()
	visibles = visibles.reshape(visibles.shape[0],784)

	learningRate = 0.01
	k = 1
	visibleSize = 784
	hiddenSize = 144
	stepNumber = 1
	
	model = RBM(learningRate, k, visibleSize, hiddenSize)
	for item in visibles:
		sess = tf.InteractiveSession()
		input = tf.placeholder(tf.float32, shape=[visibleSize], name="visible_layer")
		sess.run(tf.global_variables_initializer())
		train = model.training(input)
		#sess.run(tf.initialize_all_variables())
		sess.run(train, feed_dict = {input: item})
		# save = saveModel()
		# save.save(sess,"rbm-cd/"+str(stepNumber)+".ckpt",global_step = step)
		# print("Step %d Model Saved!",stepNumber)
		# stepNumber = stepNumber + 1