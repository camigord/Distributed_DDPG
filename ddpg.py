import tensorflow as tf
import numpy as np
from parameters import Parameters
import gym
from gym import wrappers
import tflearn
import sys
import os
import time
import itertools
from sklearn.preprocessing import StandardScaler

from networks import ActorNetwork, CriticNetwork
from replay_buffer import ReplayBuffer

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

def UONoise():
    theta = 0.15
    sigma = 0.2
    state = 0
    while True:
        yield state
        state += -theta*state+sigma*np.random.randn()

# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries():
    training_summaries = []
    episode_reward = tf.Variable(0.)
    training_summaries.append(tf.summary.scalar("Reward", episode_reward))
    episode_ave_max_q = tf.Variable(0.)
    training_summaries.append(tf.summary.scalar("Qmax Value", episode_ave_max_q))
    value_loss = tf.Variable(0.)
    training_summaries.append(tf.summary.scalar("Value Loss", value_loss))

    train_ops = tf.summary.merge(training_summaries)

    # Validation variables
    valid_summaries = []
    valid_Reward = tf.Variable(0.)
    valid_summaries.append(tf.summary.scalar("Validation Rewards", valid_Reward))

    valid_ops = tf.summary.merge(valid_summaries)

    valid_vars = [valid_Reward]
    training_vars = [episode_reward, episode_ave_max_q, value_loss]

    return train_ops, valid_ops, training_vars, valid_vars

# ===========================
#   Agent Training
# ===========================
def train(sess, current_step, opt, env, actor, critic, train_ops, training_vars, replay_buffer, writer, is_chief):
    noise = UONoise()
    state = env.reset()

    ep_reward = 0.0
    ep_ave_max_q = 0.0
    value_loss = 0.0

    for t in itertools.count():
        # Added exploration noise
        input_s = np.reshape(state, (1, actor.s_dim))
        a = actor.predict(input_s)
        a = actor.predict(input_s) + (1. / (1. + current_step))
        '''
        if current_step  < opt.max_exploration_episodes:
            p = current_step/opt.max_exploration_episodes
            a = a*p + (1-p)*next(noise)
        '''

        state2, r, done, info = env.step(a[0])

        replay_buffer.add(np.reshape(state, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r, done, np.reshape(state2, (actor.s_dim,)))

        state = state2
        ep_reward += r

        ## UPDATE NETWORK
        # Keep adding experience to the memory until there are at least minibatch size samples
        if replay_buffer.size() > opt.batch_size:
            s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(opt.batch_size)

            # Calculate targets
            target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

            y_i = []
            for k in range(opt.batch_size):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + opt.gamma * target_q[k])

            # Update the critic given the targets
            predicted_q_value, _, v_loss = critic.train(s_batch, a_batch, np.reshape(y_i, (opt.batch_size, 1)))

            ep_ave_max_q += np.amax(predicted_q_value)
            value_loss += v_loss

            # Update the actor policy using the sampled gradient
            a_outs = actor.predict(s_batch)
            grads = critic.action_gradients(s_batch, a_outs)

            actor.train(s_batch, grads[0])

            # Update target networks
            actor.update_target_network()
            critic.update_target_network()

        if done:
            break

    if is_chief:
        summary_str = sess.run(train_ops, feed_dict={
            training_vars[0]: ep_reward,
            training_vars[1]: ep_ave_max_q / float(t),
            training_vars[2]: value_loss / float(t)
        })
        writer.add_summary(summary_str, current_step)
        writer.flush()

    print('Episode: %d - Iterations: %d - Reward: %f' % (current_step, t, ep_reward))

    return ep_reward

def test(sess, current_step, opt, env, actor, critic, valid_ops, valid_vars, writer):
    valid_r = 0
    state = env.reset()

    for t in itertools.count():
        input_s = np.reshape(state, (1, actor.s_dim))
        a = actor.predict_target(input_s)

        state2, r, done, _ = env.step(a[0])
        valid_r += r

        state = state2

        if done:
            break

    summary_valid = sess.run(valid_ops, feed_dict={
        valid_vars[0]: valid_r
    })
    writer.add_summary(summary_valid, current_step)
    writer.flush()

    return valid_r

def save_model(sess, saver, opt, global_step):
    save_path = saver.save(sess, opt.save_dir + "/model", global_step=global_step)
    print('-------------------------------------')
    print("Model saved in file: %s" % save_path)
    print('-------------------------------------')


def main(_):
    opt = Parameters()
    np.random.seed(opt.seed)
    tf.set_random_seed(opt.seed)

    if opt.train:
        cluster = tf.train.ClusterSpec({"ps":opt.parameter_servers, "worker":opt.workers})
        server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

        if FLAGS.job_name == "ps":
            server.join()
        elif FLAGS.job_name == "worker":
            with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):
                is_chief = (FLAGS.task_index == 0)
                # count the number of updates
                global_step = tf.get_variable('global_step',[],initializer = tf.constant_initializer(0),trainable = False)
                step_op = global_step.assign(global_step+1)

                env = gym.make(opt.env_name)
                if is_chief:
                    env = wrappers.Monitor(env,'./tmp/',force=True)

                if opt.env_name == 'MountainCarContinuous-v0':
                    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
                    scaler = StandardScaler()
                    scaler.fit(observation_examples)
                else:
                    scaler = None

                # Initialize replay memory
                replay_buffer = ReplayBuffer(opt.rm_size, opt.seed)

                state_dim = env.observation_space.shape[0]
                action_dim = env.action_space.shape[0]
                if abs(env.action_space.low[0]) == abs(env.action_space.high[0]):
                    action_scale = abs(env.action_space.high[0])
                else:
                    print('Error: Action space in current environment is asymmetric! ')
                    sys.exit()

                actor = ActorNetwork(state_dim, action_dim, action_scale, opt.actor_lr, opt.tau, scaler)
                critic = CriticNetwork(state_dim, action_dim, opt.critic_lr , opt.tau, actor.get_num_trainable_vars(), scaler)

                # Set up summary Ops
                train_ops, valid_ops, training_vars, valid_vars = build_summaries()

                init_op = tf.global_variables_initializer()

                # Add ops to save and restore all the variables.
                saver = tf.train.Saver(max_to_keep=5)

                if opt.continue_training:
                    def restore_model(sess):
                        actor.set_session(sess)
                        critic.set_session(sess)
                        saver.restore(sess,tf.train.latest_checkpoint(opt.save_dir+'/'))
                        actor.restore_params(tf.trainable_variables())
                        critic.restore_params(tf.trainable_variables())
                        print('***********************')
                        print('Model Restored')
                        print('***********************')
                else:
                    def restore_model(sess):
                        actor.set_session(sess)
                        critic.set_session(sess)
                        # Initialize target network weights
                        actor.update_target_network()
                        critic.update_target_network()
                        print('***********************')
                        print('Model Initialized')
                        print('***********************')

                #sv = tf.train.Supervisor(is_chief=is_chief, global_step=global_step, init_op=init_op, summary_op=None, saver=None, init_fn=restore_model)

                #with sv.prepare_or_wait_for_session(server.target) as sess:
                with tf.Session(server.target) as sess:
                    sess.run(init_op)
                    restore_model(sess)

                    writer = tf.summary.FileWriter(opt.summary_dir, sess.graph)

                    stats = []
                    for step in range(opt.max_episodes):
                        '''
                        if sv.should_stop():
                            break
                        '''

                        current_step = sess.run(global_step)
                        # Train normally
                        reward = train(sess, current_step, opt, env, actor, critic, train_ops, training_vars, replay_buffer, writer, is_chief)
                        stats.append(reward)

                        if np.mean(stats[-100:]) > 950 and len(stats) >= 101:
                            print(np.mean(stats[-100:]))
                            print("Solved.")
                            if is_chief:
                                save_model(sess, saver, opt, global_step)
                            break

                        if is_chief and step % opt.valid_freq == opt.valid_freq-1:
                            #test_r = test(sess, current_step, opt, env, actor, critic, valid_ops, valid_vars, writer)
                            save_model(sess, saver, opt, global_step)

                        # Increase global_step
                        sess.run(step_op)

                #sv.stop()
                print('Done')


    else:       # For testing
        pass


if __name__ == '__main__':
    tf.app.run()
