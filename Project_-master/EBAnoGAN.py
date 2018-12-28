import tensorflow as tf
from utils import *
from data_utils import *
from keras import applications

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
            net = leaky(conv(X_noised, self.depth[0], 5, 2, name = "conv_1"))
            net = leaky(bn(conv(net, self.depth[1], 5, 2, name = "conv_2"), phase, "bn_2"))
            net = leaky(bn(conv(net, self.depth[2], 5, 2, name = "conv_3"), phase, "bn_3"))
            net = leaky(bn(conv(net, self.depth[3], 5, 2, name = "conv_4"), phase, "bn_4"))
            net = tf.layers.flatten(net, name = "flatten")
            mean = dense(net, self.conf.n_z, name = "mean")
            std = tf.nn.softplus(dense(net, self.conf.n_z, name = "std")) + 1e-6

        return mean, std

    def bernoulli_decoder(self, Z, phase):

        with tf.variable_scope("bernoulli_decoder", reuse = tf.AUTO_REUSE):
            net = dense(Z, self.depth[3]*4*4, name = "fc_1")
            net = tf.reshape(net, [self.conf.batch_size, 4, 4, self.depth[3]])
            net = tf.nn.relu(bn(net, phase, "bn_1"))
            net = relu(bn(deconv(net, self.depth[2], 5, 2, name="dconv_1"), phase, "bn_2"))
            net = relu(bn(deconv(net, self.depth[1], 5, 2, name="dconv_2"), phase, "bn_3"))
            net = relu(bn(deconv(net, self.depth[0], 5, 2, name="dconv_3"), phase, "bn_4"))
            logits = deconv(net,self.c, 5,2, name = "dconv_4")
            X_out = tf.nn.sigmoid(logits)

        return logits, X_out

    def EBGAN_PT(self,latent):

        eps = 1e-8
        latent = tf.layers.flatten(latent, name = "flatten") # [batch_size, N_latent]
        lt_norm = tf.reduce_sum(latent, axis = 1, keep_dims = True) # [batch_size,1]
        Numer = tf.matmul(latent, latent, transpose_b=True) # [batch_size, batch_size]
        Denom = tf.matmul(lt_norm, lt_norm, transpose_b = True) # [batch_size, batch_size]
        unit = tf.square(Numer/(Denom + eps))

        filter = np.ones([self.conf.batch_size, self.conf.batch_size])
        filter = change_diag_value(filter, 0)
        filter = tf.convert_to_tensor(filter, dtype = tf.float32)

        pt_loss = tf.reduce_sum(tf.multiply(unit, filter))/(self.conf.batch_size*(self.conf.batch_size-1))

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
            X_out = tf.nn.sigmoid(net)

        return X_out

    def Revised_EBGAN(self, X, X_noised, phase):

        ### perceptual VAE, Generator of GANs ########################
        mean, std = self.gaussian_encoder(X_noised, phase)
        z = mean + std * tf.random_normal(tf.shape(mean, out_type=tf.int32), 0, 1, dtype=tf.float32)
        z_c = tf.reduce_mean(z, axis = 0)
        logits, X_out = self.bernoulli_decoder(z, phase)
        X_out = tf.clip_by_value(X_out, 1e-8, 1 - 1e-8)
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

        act_X_1 = model.layers[0](X)
        act_X_2 = model.layers[1](act_X_1)
        act_X_3 = model.layers[2](act_X_2)

        p_loss_1 = tf.reduce_mean(tf.squared_difference(act_1, act_X_1))
        p_loss_2 = tf.reduce_mean(tf.squared_difference(act_2, act_X_2))
        p_loss_3 = tf.reduce_mean(tf.squared_difference(act_3, act_X_3))
        Perceptual_loss = p_loss_1 + p_loss_2 + p_loss_3

        ##############################################################

        D_latent_fake = self.EB_Discriminator_En(X_out, phase)
        D_fake = self.EB_Discriminator_De(D_latent_fake, phase)
        D_real = self.EB_Discriminator_De(self.EB_Discriminator_En(X,phase), phase)

        fake_loss = tf.reduce_mean(tf.squared_difference(D_fake, X_out))
        real_loss = tf.reduce_mean(tf.squared_difference(D_real, X))

        D_loss = real_loss + relu(self.conf.margin - fake_loss)
        G_loss = self.conf.KL_rate*KL_Div + self.conf.PL_rate*Perceptual_loss + self.conf.E_rate*Potential_loss + \
                 self.conf.pt_rate*self.EBGAN_PT(D_latent_fake) + self.conf.FL_rate*fake_loss

        return X_out, D_loss, G_loss