import tensorflow as tf
from utils import *
from data_utils import *
from keras import applications
import pdb

tf.set_random_seed(777)

class EBAnoGAN:
    def __init__(self, conf, shape, depth):
        self.conf = conf
        self.data = conf.data
        self.batch_size = conf.batch_size
        self.w = shape[1]
        self.h = shape[2]
        self.c = shape[3]
        self.length = (self.w)*(self.h)*(self.c)
        self.depth = depth

    def gaussian_encoder(self, X_noised, phase):

        with tf.variable_scope("gaussian_encoder", reuse = tf.AUTO_REUSE):
            net = leaky(conv(X_noised, self.depth[0], 3, 2, name = "conv_1"))
            net = leaky(bn(conv(net, self.depth[1], 3, 2, name = "conv_2"), phase, "bn_2"))
            net = leaky(bn(conv(net, self.depth[2], 3, 2, name = "conv_3"), phase, "bn_3"))
            net = leaky(bn(conv(net, self.depth[3], 3, 2, name = "conv_4"), phase, "bn_4"))
            net = tf.layers.flatten(net, name = "flatten")
            mean = dense(net, self.conf.n_z, name = "mean")
            std = tf.nn.softplus(dense(net, self.conf.n_z, name = "std")) + 1e-6

        return mean, std

    def bernoulli_decoder(self, Z, phase):

        with tf.variable_scope("bernoulli_decoder", reuse = tf.AUTO_REUSE):
            net = dense(Z, self.depth[3]*4*4, name = "fc_1")
            net = tf.reshape(net, [self.conf.batch_size, 4, 4, self.depth[3]])
            net = tf.nn.relu(bn(net, phase, "bn_1"))
            net = relu(bn(deconv(net, self.depth[2], 3, 2, name="dconv_1"), phase, "bn_2"))
            net = relu(bn(deconv(net, self.depth[1], 3, 2, name="dconv_2"), phase, "bn_3"))
            net = relu(bn(deconv(net, self.depth[0], 3, 2, name="dconv_3"), phase, "bn_4"))
            logits = deconv(net,self.c, 3,2, name = "dconv_4")
            X_out = tf.nn.tanh(logits)

        return logits, X_out

    def BEGAN_Dis(self,X, phase):

        with tf.variable_scope("BE_encoder", reuse = tf.AUTO_REUSE):
            net = leaky(conv(X, self.depth[0], 3, 2, name = "conv_1"))
            net = leaky(bn(conv(net, self.depth[1], 3, 2, name = "conv_2"), phase, "bn_1"))
            net = leaky(bn(conv(net, self.depth[2], 3, 2, name = "conv_3"), phase, "bn_2"))
            net = leaky(bn(conv(net, self.depth[3], 3, 2, name = "conv_4"), phase, "bn_3"))
            latent = tf.layers.flatten(net)

        with tf.variable_scope("BE_decoder", reuse = tf.AUTO_REUSE):
            net = relu(bn(deconv(latent, self.depth[2], 3, 2, name="dconv_1"), phase, "bn_1"))
            net = relu(bn(deconv(net, self.depth[1], 3, 2, name="dconv_2"), phase, "bn_2"))
            net = relu(bn(deconv(net, self.depth[0], 3, 2, name="dconv_3"), phase, "bn_3"))
            net = deconv(net,self.c, 3,2, name = "dconv_4")
            X_out = tf.nn.tanh(net)

        return latent, X_out

    def BEAnoGAN(self, X, X_noised, phase):

        ### perceptual VAE, Generator of GANs ########################
        mean, std = self.gaussian_encoder(X_noised, phase)
        z = mean + std * tf.random_normal(tf.shape(mean, out_type=tf.int32), 0, 1, dtype=tf.float32)
        z_c = tf.reduce_mean(z, axis = 0)
        logits, X_out = self.bernoulli_decoder(z, phase)

        if self.c == 1:
            X_out = tf.image.grayscale_to_rgb(X_out)
            X = tf.image.grayscale_to_rgb(X)

        KL_Div = 0.5*tf.reduce_mean(1 - tf.log(tf.square(std) + 1e-8) + tf.square(mean) + tf.square(std))
        Potential_loss = tf.reduce_mean(tf.square(z_c - z))

        model = applications.VGG19(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
        for layer in model.layers[:]:
            layer.trainable = False

        act_1 = model.layers[0](X_out)
        act_2 = model.layers[1](act_1)
        act_3 = model.layers[2](act_2)
        act_4 = model.layers[3](act_3)
        act_5 = model.layers[4](act_4)
        act_6 = model.layers[5](act_5)

        act_X_1 = model.layers[0](X)
        act_X_2 = model.layers[1](act_X_1)
        act_X_3 = model.layers[2](act_X_2)
        act_X_4 = model.layers[3](act_X_3)
        act_X_5 = model.layers[4](act_X_4)
        act_X_6 = model.layers[5](act_X_5)

        p_loss_4 = tf.reduce_mean(tf.squared_difference(act_4, act_X_4))
        p_loss_5 = tf.reduce_mean(tf.squared_difference(act_5, act_X_5))
        p_loss_6 = tf.reduce_mean(tf.squared_difference(act_6, act_X_6))

        Perceptual_loss = p_loss_4 + p_loss_5 + p_loss_6

        ##############################################################

        if self.c == 1:
            X_out = tf.image.rgb_to_grayscale(X_out)
            X = tf.image.rgb_to_grayscale(X)

        k = tf.Variable(0, trainable = False)

        fake_code, fake_img = self.BEGAN_Dis(X_out, phase)
        real_code, real_img = self.BEGAN_Dis(X, phase)

        fake_err = tf.sqrt(2*tf.nn.l2_loss(fake_img - X_out)) / self.batch_size
        real_err = tf.sqrt(2*tf.nn.l2_loss(real_img - X)) / self.batch_size

        D_loss = real_err - k*fake_err
        G_loss = fake_err

        M = real_err + tf.abs(self.conf.gamma*real_err - fake_err)
        update_k = k.assign(tf.clip_by_value(k + self.conf.lamda*(self.conf.gamma*real_err - fake_err), 0, 1))

        G_loss_VAE = self.conf.KL_rate*KL_Div + self.conf.PL_rate*Perceptual_loss + self.conf.E_rate*Potential_loss
        G_loss_GAN = self.conf.pt_rate*self.EBGAN_PT(D_latent_fake) + self.conf.FL_rate*fake_loss



        return X_out, D_loss, G_loss_VAE, G_loss_GAN, D_real, D_fake