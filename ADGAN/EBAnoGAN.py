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

    def Cycle_En(self,X, phase):

        with tf.variable_scope("Cycle_encoder", reuse = tf.AUTO_REUSE):
            net = leaky(conv(X, self.depth[0], 3, 2, name = "conv_1"))
            net = leaky(bn(conv(net, self.depth[1], 3, 2, name = "conv_2"), phase, "bn_1"))
            net = leaky(bn(conv(net, self.depth[2], 3, 2, name = "conv_3"), phase, "bn_2"))
            net = leaky(bn(conv(net, self.depth[3], 3, 2, name = "conv_4"), phase, "bn_3"))
            net = tf.layers.flatten(net)
            latent = dense(net, self.conf.n_z, name = "latent_vector")

        return latent

    def Cycle_De(self, latent, phase):

        with tf.variable_scope("Cycle_decoder", reuse = tf.AUTO_REUSE):
            net = dense(latent, self.depth[3] * 4 * 4, name="fc_1")
            net = tf.reshape(net, [self.conf.batch_size, 4, 4, self.depth[3]])
            net = tf.nn.relu(bn(net, phase, "bn_1"))
            net = relu(bn(deconv(net, self.depth[2], 3, 2, name="dconv_1"), phase, "bn_2"))
            net = relu(bn(deconv(net, self.depth[1], 3, 2, name="dconv_2"), phase, "bn_3"))
            net = relu(bn(deconv(net, self.depth[0], 3, 2, name="dconv_3"), phase, "bn_4"))
            net = deconv(net,self.c, 3,2, name = "dconv_4")
            X_out = tf.nn.tanh(net)

        return X_out

    # borrowed from https://github.com/shekkizh/EBGAN.tensorflow/blob/master/EBGAN/Faces_EBGAN.py
    def EBGAN_PT(self,embeddings):

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        similarity = tf.matmul(
            normalized_embeddings, normalized_embeddings, transpose_b=True)
        batch_size = tf.cast(tf.shape(embeddings)[0], tf.float32)
        pt_loss = (tf.reduce_sum(similarity) - batch_size) / (batch_size * (batch_size - 1))

        return pt_loss

    def EB_Discriminator_En(self,X, phase):

        with tf.variable_scope("EB_encoder", reuse = tf.AUTO_REUSE):
            net = leaky(conv(X, self.depth[0], 5, 2, name = "conv_1"))
            net = leaky(bn(conv(net, self.depth[1], 5, 2, name = "conv_2"), phase, "bn_1"))
            net = leaky(bn(conv(net, self.depth[2], 5, 2, name = "conv_3"), phase, "bn_2"))
            net = leaky(bn(conv(net, self.depth[3], 5, 2, name = "conv_4"), phase, "bn_3"))

        return net

    def EB_Discriminator_De(self, feature_maps, phase):

        with tf.variable_scope("EB_decoder", reuse = tf.AUTO_REUSE):
            net = relu(bn(deconv(feature_maps, self.depth[2], 5, 2, name="dconv_1"), phase, "bn_1"))
            net = relu(bn(deconv(net, self.depth[1], 5, 2, name="dconv_2"), phase, "bn_2"))
            net = relu(bn(deconv(net, self.depth[0], 5, 2, name="dconv_3"), phase, "bn_3"))
            net = deconv(net,self.c, 5,2, name = "dconv_4")
            X_out = tf.nn.tanh(net)

        return X_out

    def ADGAN(self, X, X_noised, phase):

        ### perceptual VAE, G of ADGAN ############################
        mean, std = self.gaussian_encoder(X_noised, phase)
        z_r2f = mean + std * tf.random_normal(tf.shape(mean, out_type=tf.int32), 0, 1, dtype=tf.float32)
        f_logits, f_out = self.bernoulli_decoder(z_r2f, phase)

        if self.c == 1:
            f_out = tf.image.grayscale_to_rgb(f_out)
            X = tf.image.grayscale_to_rgb(X)

        model = applications.VGG19(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
        for layer in model.layers[:]:
            layer.trainable = False

        act_1 = model.layers[0](f_out)
        act_2 = model.layers[1](act_1)
        act_3 = model.layers[2](act_2)

        act_X_1 = model.layers[0](X)
        act_X_2 = model.layers[1](act_X_1)
        act_X_3 = model.layers[2](act_X_2)

        p_loss_1 = tf.reduce_mean(tf.squared_difference(act_1, act_X_1))
        p_loss_2 = tf.reduce_mean(tf.squared_difference(act_2, act_X_2))
        p_loss_3 = tf.reduce_mean(tf.squared_difference(act_3, act_X_3))

        if self.c == 1:
            f_out = tf.image.rgb_to_grayscale(f_out)
            X = tf.image.rgb_to_grayscale(X)

        KL_Div = 0.5 * tf.reduce_mean(1 - tf.log(tf.square(std) + 1e-8) + tf.square(mean) + tf.square(std))
        Perceptual_loss = p_loss_1 + p_loss_2 + p_loss_3

        ori2fake_loss = Perceptual_loss + self.conf.KL_rate*KL_Div
        ##############################################################

        ### Plain AutoEncoder, F of ADGAN ############################
        z_f2r = self.Cycle_En(f_out, phase)
        r_out = self.Cycle_De(z_f2r, phase)

        fake2ori_loss = tf.reduce_mean(tf.squared_difference(r_out,X))
        ##############################################################

        ### EB Discriminator, Discriminator of ADGAN #################
        D_fake_latent = self.EB_Discriminator_En(r_out, phase)
        D_fake = self.EB_Discriminator_De(D_fake_latent, phase)
        D_real = self.EB_Discriminator_De(self.EB_Discriminator_En(X, phase), phase)

        fake_loss = tf.reduce_mean(tf.squared_difference(D_fake, r_out))
        real_loss = tf.reduce_mean(tf.squared_difference(D_real, X))
        D_loss = real_loss + relu(self.conf.margin - fake_loss)
        PT_loss = self.EBGAN_PT(D_fake_latent)
        G_loss = self.conf.PVAE_rate*ori2fake_loss + self.conf.AE_rate*fake2ori_loss +\
                 self.conf.PT_rate*PT_loss + self.conf.DIS_rate*real_loss
        ##############################################################

        return z_r2f, f_out, ori2fake_loss, z_f2r, r_out, fake2ori_loss, self.EBGAN_PT(D_fake_latent), real_loss, G_loss, D_loss
