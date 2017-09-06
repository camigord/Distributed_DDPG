import tensorflow as tf
import tflearn

# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -2 and 2
    """

    def __init__(self,  state_dim, action_dim, action_scale, learning_rate, tau, scaler):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.scaler = scaler
        self.action_scale = action_scale

        # Actor Network
        self.inputs, self.out = self.create_actor_network()
        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out = self.create_actor_network()
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        # Partial derivatives of out w.r.t network_params. action_gradient holds the initial gradients for each out
        self.actor_gradients = tf.gradients(self.out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])

        layer1 = tflearn.fully_connected(inputs, 128, activation='elu', regularizer='L2', weight_decay=0.1)
        layer2 = tflearn.fully_connected(layer1, 200, activation='elu', regularizer='L2', weight_decay=0.1)

        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(layer2, self.a_dim, activation='tanh', bias=False, weights_init=w_init, regularizer='L2')

        scaled_out = tf.multiply(out, self.action_scale)

        return inputs, scaled_out

    def preprocess_input(self, inputs):
        if self.scaler != None:
            return self.scaler.transform(inputs)
        else:
            return inputs

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: self.preprocess_input(inputs),
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: self.preprocess_input(inputs)
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: self.preprocess_input(inputs)
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def restore_params(self, parameters):
        restore_network = [self.network_params[i].assign(parameters[i]) for i in range(len(self.network_params))]
        restore_target = [self.target_network_params[i].assign(parameters[i+len(self.network_params)]) for i in range(len(self.target_network_params))]
        self.sess.run([restore_network, restore_target])

    def set_session(self,sess):
        self.sess = sess


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, state_dim, action_dim, learning_rate, tau, num_actor_vars, scaler):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.num_actor_vars = num_actor_vars
        self.scaler = scaler

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()
        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        #self.clipped_value = tf.clip_by_value(self.predicted_q_value, -100.0, 0.0)

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the network w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])

        net_state = tflearn.fully_connected(inputs, 128, activation='elu', regularizer='L2', weight_decay=0.1)
        net_action = tflearn.fully_connected(action, 128, activation='elu', regularizer='L2', weight_decay=0.1)

        hidden = tf.concat([net_state,net_action],axis=1)

        net = tflearn.fully_connected(hidden, 200, activation='elu', regularizer='L2', weight_decay=0.1)

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init, regularizer='L2')
        return inputs, action, out

    def preprocess_input(self, inputs):
        if self.scaler != None:
            return self.scaler.transform(inputs)
        else:
            return inputs

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize, self.loss], feed_dict={
            self.inputs: self.preprocess_input(inputs),
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: self.preprocess_input(inputs),
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: self.preprocess_input(inputs),
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: self.preprocess_input(inputs),
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def restore_params(self, parameters):
        restore_network = [self.network_params[i].assign(parameters[i+self.num_actor_vars]) for i in range(len(self.network_params))]
        restore_target = [self.target_network_params[i].assign(parameters[i+self.num_actor_vars+len(self.network_params)]) for i in range(len(self.target_network_params))]
        self.sess.run([restore_network, restore_target])

    def set_session(self,sess):
        self.sess = sess
