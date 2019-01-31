import tensorflow as tf
from utils import *
from data_utils import *
from keras import applications

tf.set_random_seed(777)

class PVAE:
    def __init__(self, conf, shape, depth):
        self.conf = conf
        self.data = conf.data
        self.batch_size = conf.batch_size
        self.h = shape[1]
        self.w = shape[2]
        self.c = shape[3]
        self.length = (self.h)*(self.w)*(self.c)
        self.depth = depth

    def gaussian_encoder(self, X_noised, phase):

        with tf.variable_scope("Gaussian_encoder", reuse = tf.AUTO_REUSE):
            net = leaky(conv(X_noised, self.depth[0], 3, 2, name = "conv_1"))
            net = leaky(bn(conv(net, self.depth[1], 3, 2, name = "conv_2"), phase, "bn_2"))
            net = leaky(bn(conv(net, self.depth[2], 3, 2, name = "conv_3"), phase, "bn_3"))
            net = leaky(bn(conv(net, self.depth[3], 3, 2, name = "conv_4"), phase, "bn_4"))
            self.acts_gH = net.get_shape().as_list()[1]
            self.acts_gW = net.get_shape().as_list()[2]
            net = tf.layers.flatten(net, name = "flatten")
            mean = dense(net, self.conf.n_z, name = "mean")
            std = tf.nn.softplus(dense(net, self.conf.n_z, name = "std")) + 1e-6

        return mean, std

    def bernoulli_decoder(self, Z, phase):

        with tf.variable_scope("Bernoulli_decoder", reuse = tf.AUTO_REUSE):
            net = dense(Z, self.depth[3]*self.acts_gW*self.acts_gH, name = "fc_1")
            net = tf.reshape(net, [self.conf.batch_size, self.acts_gH, self.acts_gW, self.depth[3]])
            net = tf.nn.relu(bn(net, phase, "bn_1"))
            net = relu(bn(deconv(net, self.depth[2], 3, 2, name="dconv_1"), phase, "bn_2"))
            net = relu(bn(deconv(net, self.depth[1], 3, 2, name="dconv_2"), phase, "bn_3"))
            net = relu(bn(deconv(net, self.depth[0], 3, 2, name="dconv_3"), phase, "bn_4"))
            net = deconv(net,self.c, 3,2, name = "dconv_4")
            X_out = tf.nn.tanh(net)

        return X_out

    def build_model(self, X, X_noised, phase):

        ### 1.PVAE Generator ###

        model = applications.VGG19(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
        for layer in model.layers[:]:
            layer.trainable = False

        mean, std = self.gaussian_encoder(X_noised, phase)
        s_latent = mean + std * tf.random_normal(tf.shape(mean, out_type=tf.int32), 0, 1, dtype=tf.float32)
        s_out = self.bernoulli_decoder(s_latent, phase)

        act_1 = model.layers[0](s_out)
        act_2 = model.layers[1](act_1)
        act_3 = model.layers[2](act_2)

        act_X_1 = model.layers[0](X)
        act_X_2 = model.layers[1](act_X_1)
        act_X_3 = model.layers[2](act_X_2)

        p_loss_1 = tf.reduce_mean(tf.squared_difference(act_1, act_X_1))
        p_loss_2 = tf.reduce_mean(tf.squared_difference(act_2, act_X_2))
        p_loss_3 = tf.reduce_mean(tf.squared_difference(act_3, act_X_3))

        Recon_loss = p_loss_1 + p_loss_2 + p_loss_3

        KL_Div = 0.5 * tf.reduce_mean(1 - tf.log(tf.square(std) + 1e-8) + tf.square(mean) + tf.square(std))
        s_loss = Recon_loss + KL_Div
        
        return s_loss
        
