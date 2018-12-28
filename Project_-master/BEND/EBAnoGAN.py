import tensorflow as tf
from utils import *
from data_utils import *
from keras import applications
import pdb

tf.set_random_seed(777)

class ADGAN:
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

    def Cycle_En(self,X, phase):

        with tf.variable_scope("Cycle_encoder", reuse = tf.AUTO_REUSE):
            net = leaky(conv(X, self.depth[0], 3, 2, name = "conv_1"))
            net = leaky(bn(conv(net, self.depth[1], 3, 2, name = "conv_2"), phase, "bn_1"))
            net = leaky(bn(conv(net, self.depth[2], 3, 2, name = "conv_3"), phase, "bn_2"))
            net = leaky(bn(conv(net, self.depth[3], 3, 2, name = "conv_4"), phase, "bn_3"))
            self.acts_cH = net.get_shape().as_list()[1]
            self.acts_cW = net.get_shape().as_list()[2]
            net = tf.layers.flatten(net)
            latent = dense(net, self.conf.n_z, name = "latent_vector")

        return latent

    def Cycle_De(self, latent, phase):

        with tf.variable_scope("Cycle_decoder", reuse = tf.AUTO_REUSE):
            net = dense(latent, self.depth[3] * self.acts_cW * self.acts_cH, name="fc_1")
            net = tf.reshape(net, [self.conf.batch_size, self.acts_cH, self.acts_cW, self.depth[3]])
            net = tf.nn.relu(bn(net, phase, "bn_1"))
            net = relu(bn(deconv(net, self.depth[2], 3, 2, name="dconv_1"), phase, "bn_2"))
            net = relu(bn(deconv(net, self.depth[1], 3, 2, name="dconv_2"), phase, "bn_3"))
            net = relu(bn(deconv(net, self.depth[0], 3, 2, name="dconv_3"), phase, "bn_4"))
            net = deconv(net,self.c, 3,2, name = "dconv_4")
            X_out = tf.nn.tanh(net)

        return X_out

    def BEGAN_Dis(self,X, phase):

        with tf.variable_scope("BE_encoder", reuse = tf.AUTO_REUSE):
            net = leaky(conv(X, self.depth[0], 3, 2, name = "conv_1"))
            net = leaky(bn(conv(net, self.depth[1], 3, 2, name = "conv_2"), phase, "bn_1"))
            net = leaky(bn(conv(net, self.depth[2], 3, 2, name = "conv_3"), phase, "bn_2"))
            self.acts_batch = net.get_shape().as_list()[0]
            self.acts_dH = net.get_shape().as_list()[1]
            self.acts_dW = net.get_shape().as_list()[2]
            net = tf.layers.flatten(net)
            latent = dense(net, self.conf.n_z, name = "fc1")

        with tf.variable_scope("BE_decoder", reuse = tf.AUTO_REUSE):
            net = dense(latent, self.depth[2] * self.acts_dW * self.acts_dH, name = "fc_1")
            net = tf.reshape(net, [self.acts_batch, self.acts_dH, self.acts_dW, self.depth[2]])
            net = tf.nn.relu(bn(net, phase, "bn_1"))
            net = relu(bn(deconv(net, self.depth[1], 3, 2, name="dconv_1"), phase, "bn_2"))
            net = relu(bn(deconv(net, self.depth[0], 3, 2, name="dconv_2"), phase, "bn_3"))
            net = deconv(net,self.c, 3,2, name = "dconv_3")
            X_out = tf.nn.tanh(net)

        return X_out

    def ADGAN(self, X, X_noised, phase):

        ### 1.PVAE Generator ###

        model = applications.VGG19(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
        for layer in model.layers[:]:
            layer.trainable = False

        mean, std = self.gaussian_encoder(X_noised, phase)
        s_latent = mean + std * tf.random_normal(tf.shape(mean, out_type=tf.int32), 0, 1, dtype=tf.float32)
        s_out = self.bernoulli_decoder(s_latent, phase)
        Heat_map_1 = make_l2(X, s_out)

        if self.conf.PVAE is True:
            if self.c == 1:
                s_out = tf.image.grayscale_to_rgb(s_out)
                X = tf.image.grayscale_to_rgb(X)

            act_1 = model.layers[0](s_out)
            act_2 = model.layers[1](act_1)
            act_3 = model.layers[2](act_2)

            act_X_1 = model.layers[0](X)
            act_X_2 = model.layers[1](act_X_1)
            act_X_3 = model.layers[2](act_X_2)


            p_loss_1 = tf.reduce_mean(tf.squared_difference(act_1, act_X_1))
            p_loss_2 = tf.reduce_mean(tf.squared_difference(act_2, act_X_2))
            p_loss_3 = tf.reduce_mean(tf.squared_difference(act_3, act_X_3))

            if self.c == 1:
                s_out = tf.image.rgb_to_grayscale(s_out)
                X = tf.image.rgb_to_grayscale(X)

            Recon_loss = p_loss_1 + p_loss_2 + p_loss_3

        else:
            Recon_loss = tf.reduce_mean(tf.squared_difference(s_out, X))

        KL_Div = 0.5 * tf.reduce_mean(1 - tf.log(tf.square(std) + 1e-8) + tf.square(mean) + tf.square(std))
        s_loss = self.conf.PVAE_rate*(self.conf.KL_rate*KL_Div + Recon_loss)

        ######


        ### Plain AutoEncoder, G of ADGAN ###

        g_latent = self.Cycle_En(s_out, phase)
        g_out = self.Cycle_De(g_latent, phase)
        cycle_loss = self.conf.Cycle_lamda*tf.reduce_mean(tf.squared_difference(g_out, X))
        Heat_map_2 = make_l2(s_out, g_out)
        ######


        ### AutoEncoder, D of ADGAN ###

        if self.conf.make_patches is False:
            X_patches = X
            g_patches = g_out
        else:
            X_patches = make_patch(X,[self.conf.batch_size,self.h,self.w,self.c], self.conf.patch_size, self.conf.patch_strides)
            g_patches = make_patch(g_out, [self.conf.batch_size,self.h,self.w,self.c], self.conf.patch_size, self.conf.patch_strides)

        k = tf.Variable(0.0, trainable=False)

        D_g = self.BEGAN_Dis(g_patches, phase)
        D_X = self.BEGAN_Dis(X_patches, phase)
        n_patches = D_g.get_shape().as_list()[0]

        Heat_map_3 = tf.squared_difference((D_g/self.conf.gamma),D_X)

        fake_err = tf.sqrt(2*tf.nn.l2_loss(g_patches - D_g)) / n_patches
        real_err = tf.sqrt(2*tf.nn.l2_loss(X_patches - D_X)) / n_patches


        D_loss = real_err - k*fake_err
        G_loss = fake_err + cycle_loss

        M = real_err + tf.abs(self.conf.gamma * real_err - fake_err)
        update_k = k.assign(tf.clip_by_value(k + self.conf.lamda * (self.conf.gamma * real_err - fake_err), 0, 1))
        Heat_map = Heat_map_1 + Heat_map_2
        Decision_value = tf.reduce_mean(Heat_map, axis = (1,2,3))
        ######

        return s_latent, s_out, s_loss, g_latent, g_out, X_patches, D_loss, G_loss, Heat_map, Decision_value, M, k, update_k