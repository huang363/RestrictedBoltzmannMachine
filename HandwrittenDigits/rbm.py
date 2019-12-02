import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow.python.keras.backend.set_session as set_session
# from PIL import Image

class RBM():

	def __init__(self, learningRate, k, visibleSize, hiddenSize):
		self.k = k	# number of gibbs sampling execution
		self.learningRate = learningRate
		self.visibleSize = visibleSize
		self.hiddenSize = hiddenSize
		self.visibleBias = tf.Variable(tf.zeros([1, self.visibleSize]),tf.float32, name="visible_bias")
		self.hiddenBias = tf.Variable(tf.zeros([1, self.hiddenSize]),tf.float32, name="hidden_bias")
		self.weight = tf.Variable(tf.random.normal([self.hiddenSize,self.visibleSize], mean=0., stddev=4 * np.sqrt(6. / (self.visibleSize + self.hiddenSize))), name="weights")

	def gibbsSampling(self,negativeVisble):
		for k in range(self.k):
			hiddens = self.pHiddenGivenInput(negativeVisble)
			negativeHidden = tf.nn.relu(tf.sign(hiddens - tf.random.uniform(tf.shape(hiddens))))
			visibles = self.pInputGivenHidden(negativeHidden)
			negativeVisble = tf.nn.relu(tf.sign(visibles - tf.random.uniform(tf.shape(visibles))))
		return negativeVisble

	def sigm(self,x):
		return 1/(1 + tf.exp(-x))

	def pInputGivenHidden(self,hidden):
		input = tf.reshape(tf.matmul(hidden,self.weight),[1,self.visibleSize])
		return self.sigm(tf.add(self.visibleBias,input))

	def pHiddenGivenInput(self,input):
		hidden = tf.reshape(tf.matmul(self.weight,tf.reshape(input,[self.visibleSize,1])),[1,self.hiddenSize])
		return self.sigm(tf.add(self.hiddenBias,hidden))

	def updateWeight(self, negativeVisible, negativeHidden):
		self.visibleBias = self.visibleBias+tf.multiply(self.learningRate,tf.math.subtract(self.visibleLayer,negativeVisible))
		self.hiddenBias = self.hiddenBias + self.learningRate*(self.hiddenLayer - negativeHidden)
		updates =  tf.tensordot(tf.reshape(negativeHidden,[self.hiddenSize]),tf.reshape(negativeVisible,[self.visibleSize]),axes=0)
		self.weight = self.weight+tf.matmul(tf.reshape(self.hiddenLayer,[self.hiddenSize,1]),self.visibleLayer) - updates


	def training(self,input):
		self.visibleLayer = input
		self.hiddenLayer = self.pHiddenGivenInput(self.visibleLayer)

		visible = tf.identity(self.visibleLayer)
		negativeVisible = self.gibbsSampling(visible)
		negativeHidden = self.pHiddenGivenInput(negativeVisible)
		self.updateWeight(negativeVisible,negativeHidden)
		return self.weight,self.visibleBias,self.hiddenBias

	def plot(self):
		return 0

	def testSample(self, visibles, steps=5000):
		for step in range(steps):
			h = self.pHiddenGivenInput(visibles)
			hiddens = tf.nn.relu(tf.sign(h - tf.random.uniform(tf.shape(h))))
			v = self.pInputGivenHidden(hiddens)
			visibles = tf.nn.relu(tf.sign(v - tf.random.uniform(tf.shape(v))))
		return visibles

if __name__ == '__main__':
	numSteps = 1
	(visibles, classifications), (Testvisibles, testClassifications) = tf.keras.datasets.mnist.load_data()
	visibles = visibles.reshape(visibles.shape[0],784)
	Testvisibles = Testvisibles.reshape(Testvisibles.shape[0],784)
	testClassifications = testClassifications.reshape(testClassifications.shape[0])

	learningRate = 0.01
	k = 1
	visibleSize = 784
	hiddenSize = 500
	stepNumber = 1
	epochs = 5
	batchsize = 200

	tf.compat.v1.disable_eager_execution()
	
	model = RBM(learningRate, k, visibleSize, hiddenSize)


	input = tf.compat.v1.placeholder(tf.float32, shape=[1,visibleSize], name="visible_layer")
	train = model.training(input)
	saver = tf.compat.v1.train.Saver()

	with tf.compat.v1.Session() as sess:
		# for epoch in range(epochs):
		# 	numBatches = int(len(visibles)/batchsize-1)
		# 	for batch in range(numBatches):
		# 		batchdata = visibles[batch*batchsize:((batch+1)*batchsize-1)]
		for itr in range(len(visibles)):
			item = visibles[itr]
			sess.run(tf.compat.v1.global_variables_initializer())
			item = np.reshape(item,(1,visibleSize))
			weight, visible_bias, hidden_bias = sess.run(train, feed_dict = {input: item})

			if itr%1000 == 0:
				save_path = saver.save(sess, "./HandwrittenDigits/checkpoints/model"+str(stepNumber)+".ckpt")
				stepNumber = stepNumber + 1

	test = model.testSample(input)
	with tf.compat.v1.Session() as sess:
		print('Test')
		sess.run(tf.compat.v1.global_variables_initializer())
		for item in range(10):
			testobj = np.reshape(Testvisibles[item],(1,visibleSize))
			testobj = testobj.astype(np.float32)
			print(testobj.shape)
			print(testobj.dtype)
			result = sess.run(test, feed_dict = {input: testobj})
			result = result.reshape([28, 28])
			plt.gray()
			plt.imshow(result)
			plt.savefig('./HandwrittenDigits/test/test-image-'+str(item)+'-'+str(testClassifications[item])+'.png')
			print('Saved samples.')
		
	print("Mission Accomplished")