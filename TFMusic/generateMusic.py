import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import midi_manipulation 


def get_songs(path):
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files):
        try:
            song = np.array(midi_manipulation.midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            raise e           
    return songs

songs = get_songs('training_Midi') #These songs have already been converted from midi to msgpack
print "{} songs processed".format(len(songs))


def sample(probs):
    #Takes in a vector of probabilities, and returns a random vector of 0s and 1s sampled from the input vector
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

#This function runs the gibbs chain. We will call this function in two places:
#    - When we define the training update step
#    - When we sample our music segments from the trained RBM
def gibbs_sample(k):
    #Runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv
    def gibbs_step(count, k, xk):
        #Runs a single gibbs step. The visible values are initialized to xk
        hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh)) #Propagate the visible values to sample the hidden values
        xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv)) #Propagate the hidden values to sample the visible values
        return count+1, k, xk

    #Run gibbs steps for k iterations
    ct = tf.constant(0) #counter
    [_, _, x_sample] = control_flow_ops.While(lambda count, num_iter, *args: count < num_iter,
                                         gibbs_step, [ct, tf.constant(k), x], 1, False)
    #This is not strictly necessary in this implementation, but if you want to adapt this code to use one of TensorFlow's
    #optimizers, you need this in order to stop tensorflow from propagating gradients back through the gibbs step
    x_sample = tf.stop_gradient(x_sample) 
    return x_sample


low_note = midi_manipulation.lowerBound
high_note = midi_manipulation.upperBound
note_range = high_note - low_note

timesteps = 20
visible_layer_nodes = 2*note_range*timesteps
hidden_layer_nodes = 50


num_epochs = 200
batch_size = 100
learning_rate = tf.constant(0.005, tf.float32)


inputter = tf.placeholder(tf.float32, [None, visible_layer_nodes], name="inputter")
weight = tf.Variable(tf.random_normal([visible_layer_nodes, hidden_layer_nodes],0.01),name="weight")
bias_hidden = f.Variable(tf.zeros([1, hidden_layer_nodes],tf.float32,name="bias_hidden"))
bias_visible = f.Variable(tf.zeros([1, hidden_layer_nodes],tf.float32,name="bias_visible"))

inputter_sample = gibbs_sample(1)
h = sample(tf.sigmoid(tf.matmul(inputter,weight)+bias_hidden))
h_sample = sample(tf.sigmoid(tf.matmul(inputter_sample,weight)+bias_hidden))


size_bt = tf.cast(tf.shape(inputter)[0], tf.float32)
weight_adder = tf.mul(learning_rate/size_bt, tf.sub(tf.matmul(tf.transpose(x), h), 
	tf.matmul(tf.transpose(inputter_sample), h_sample)))
bv_adder = tf.mul(learning_rate/size_bt, tf.reduce_sum(tf.sub(inputter,inputter_sample), 0, True))
bh_adder = tf.mul(learning_rate/size_bt, tf.reduce_sum(tf.sub(h,h_sample), 0, True))

update = [weight.assign_add(weight_adder), bias_visible.assign_add(bv_adder), bias_hidden.assign_add(bh_adder)]

with tf.Session() as sess:
	init = tf.initialize_all_variables()
	sess.run(init)

	for epoch in tqdm(range(num_epochs)):
		for song in songs:
			song = np.array(song)

			for i in range(1, len(song), batch_size):
				tr_x = song[i:i+batch_size]
				sess.run(update, feed_dict={x: tr_x})

	sample = gibbs_sample(1).eval(session = sess, feed_dict = {x:np.zeros((10,visible_layer_nodes))})

	for i in range(sample.shape[0]):
		if not any(sample[i:]):
			continue

		S = np.reshape(sample[i,:], (timesteps, 2*note_range))
		midi_manipulation.noteStateMatrixToMidi(S, "generated_chord_{}".format(i))
