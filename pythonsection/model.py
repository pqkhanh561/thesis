import tensorflow as tf




class model():
		def __init__(self, n_input, n_output, sess):
			self.X = tf.placeholder(tf.float32, shape = [None, n_input] , name = 'X_in')
			self.Y = tf.placeholder(tf.float32, shape = [None, n_output], name = 'Y')
			self.Y_hat = self.build_model(n_output)
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.Y_hat))		
			self.train = tf.train.AdamOptimizer(0.001).minimize(loss)
			init= tf.global_variables_initializer()
			self.sess = sess
			self.sess.run(init)

		def build_model(self, n_output):
			fc1 = tf.contrib.layers.fully_connected(self.X, 64, activation_fn=tf.nn.relu, scope='fc1')
			fc2 = tf.contrib.layers.fully_connected(fc1, 64, activation_fn=tf.nn.relu, scope='fc2')
			out = tf.contrib.layers.fully_connected(fc2, n_output, activation_fn=None, scope='out')
			return out
				
		def fit(self, x, y, epochs, verbose):
			self.sess.run(self.train, feed_dict={self.X: x, self.Y:y})

		def predict(self, x):
			return(self.sess.run(self.Y_hat, feed_dict={self.X:x}))

		def predict_on_batch(self, x):
			return(self.predict(x))

