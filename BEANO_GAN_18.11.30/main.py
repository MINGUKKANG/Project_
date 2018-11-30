from data_utils import *
from utils import *
from plot import *
from EBAnoGAN import *
import time
from numpy.linalg import inv
import pdb

DEFINE_string("data", "MNIST", "[MNIST|MNIST_Fashion|CIFAR|CALTECH]")
DEFINE_integer("n_channel", 1, "Number of channels")

DEFINE_boolean("make_patches", False, "boolean for patches GAN")
DEFINE_integer("n_epoch", 100, "number of Epoch for training")
DEFINE_integer("n_z", 128, "Dimension of Latent Variables")
DEFINE_integer("batch_size", 16, "Batch Size for training")
DEFINE_integer("pre_epoch", 5, "epochs for pretraining of PVAE")
DEFINE_integer("n_img_plot", 100, "Number of Images for plotting")
DEFINE_integer("patch_size", 32, "patch_size for EB Discriminator")
DEFINE_integer("patch_strides", 4, "patch strides for EB Discriminator")

DEFINE_float("lr", 0.00025, "learning rate for training")
DEFINE_float("KL_rate", 0.5, "Weight for Regularization loss")
DEFINE_float("PVAE_rate", 1.0, "Weight for ori2fake loss")
DEFINE_float("Cycle_lamda", 0.4, "Weight for Cycle loss")
DEFINE_float("beta1", 0.5, "")
DEFINE_float("gamma", 1.3, "Hyper parameter of BEGAN")
DEFINE_float("lamda", 0.001, "Hyper parameter of BEGAN")
conf = print_user_flags(line_limit = 100)
print("-"*80)

if conf.n_channel ==1:
    plot_type = "C_1"
else:
    plot_type = "C_3"

FLAG_1_Data = data_controller(type = conf.data,
                              n_channel = conf.n_channel,
                              enlr_size = 64,
                              normal = [0,1,2,3,4],
                              anomalus = [5,6,7,8,9],
                              num_normal_train = 15000,
                              num_normal_test = 9500,
                              num_abnormal_test = 500,
                              name = "FLAG_1")

train_data, train_labels, test_data, test_labels = FLAG_1_Data.preprocessing()
print("performing preprocessing, augmentatioin...")
noised_train_data = salt_pepper_noise(train_data)
print("")

plot_manifold_canvas(train_data[0:conf.n_img_plot],int(conf.n_img_plot**0.5), type = plot_type, make_dir = "s_img_train", name ="ori_img_train")
plot_manifold_canvas(noised_train_data[0:conf.n_img_plot],int(conf.n_img_plot**0.5), type = plot_type, make_dir = "s_img_train", name ="noised_img_train")
plot_manifold_canvas(test_data[0:conf.n_img_plot],int(conf.n_img_plot**0.5), type = plot_type, make_dir = "s_img_test", name = "ori_img_test")
plot_manifold_canvas(train_data[0:conf.n_img_plot],int(conf.n_img_plot**0.5), type = plot_type, make_dir = "g_img_train", name ="ori_img_train")
plot_manifold_canvas(noised_train_data[0:conf.n_img_plot],int(conf.n_img_plot**0.5), type = plot_type, make_dir = "g_img_train", name ="noised_img_train")
plot_manifold_canvas(test_data[0:conf.n_img_plot],int(conf.n_img_plot**0.5), type = plot_type, make_dir = "g_img_test", name = "ori_img_test")

shape = np.shape(train_data)
H = shape[1]
W = shape[2]
C = shape[3]

X = tf.placeholder(tf.float32, shape = [conf.batch_size, W,H,C], name = "inputs")
X_noised = tf.placeholder(tf.float32, shape = [conf.batch_size, W,H,C], name = "inputs_noised")
phase = tf.placeholder(tf.bool, name = "training_phase")
lr = tf.placeholder(tf.float32, name = "lr")

ADGAN = ADGAN(conf, shape, [128,256,512,1024])
s_latent, s_out, S_loss, g_latent, g_out, X_patches, D_loss, G_loss, M, K, update_k = ADGAN.ADGAN(X,X_noised,phase)

total_batch_train = FLAG_1_Data.get_total_batch(train_data, conf.batch_size)
total_batch_test = FLAG_1_Data.get_total_batch(test_data, conf.batch_size)
FLAG_1_Data.initialize_batch()

total_vars = tf.trainable_variables()

Se_vars = [var for var in total_vars if "Gaussian_encoder" in var.name]
Sd_vars = [var for var in total_vars if "Bernoulli_decoder" in var.name]
S_vars = Se_vars + Sd_vars

Ge_vars = [var for var in total_vars if "Cycle_encoder" in var.name]
Gd_vars = [var for var in total_vars if "Cycle_decoder" in var.name]
G_vars = Ge_vars + Gd_vars

De_vars = [var for var in total_vars if "BE_encoder" in var.name]
Dd_vars = [var for var in total_vars if "BE_decoder" in var.name]
D_vars = De_vars + Dd_vars


with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    op_S = tf.train.AdamOptimizer(learning_rate = conf.lr*2, beta1 = conf.beta1).minimize(S_loss, var_list = S_vars)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    op_G = tf.train.AdamOptimizer(learning_rate = conf.lr, beta1 = conf.beta1).minimize(G_loss, var_list = G_vars)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    op_D = tf.train.RMSPropOptimizer(learning_rate = conf.lr).minimize(D_loss, var_list = D_vars)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

start_time = time.time()
m_holder = []
k_holder = []

for i in range(conf.pre_epoch):
    for j in range(total_batch_train):
        batch_xs, batch_noised_xs, batch_ys = FLAG_1_Data.next_batch(train_data, noised_train_data, train_labels, conf.batch_size)
        feed_dict = {X: batch_xs, X_noised: batch_noised_xs, phase: True, lr: conf.lr}
        _ = sess.run(op_S, feed_dict = feed_dict)
FLAG_1_Data.initialize_batch()

for i in range(conf.n_epoch):
    l_s = 0
    l_d = 0
    l_g = 0
    for j in range(total_batch_train):
        batch_xs, batch_noised_xs, batch_ys = FLAG_1_Data.next_batch(train_data, noised_train_data,train_labels ,conf.batch_size)
        feed_dict= {X: batch_xs, X_noised: batch_noised_xs, phase: True, lr: conf.lr}
        _, s = sess.run([op_S, S_loss], feed_dict = feed_dict)
        _, d = sess.run([op_D, D_loss], feed_dict = feed_dict)
        _, g = sess.run([op_G, G_loss], feed_dict = feed_dict)
        _, M_value, K_value = sess.run([update_k, M, K], feed_dict =feed_dict)
        l_s += s/total_batch_train
        l_d += d/total_batch_train
        l_g += g/total_batch_train

    m_holder.append(M_value)
    k_holder.append(K_value)

    if i % 1 == 0 or i ==(conf.n_epoch -1):
        s_train_name = "s_img_train_" + str(i)
        s_test_name = "s_img_test_" + str(i)
        g_train_name = "g_img_train" + str(i)
        g_test_name = "g_img_test" + str(i)

        if conf.make_patches is True:
            feed_dict = {X: train_data[0: conf.batch_size]}
            patch = sess.run(X_patches, feed_dict = feed_dict)
            plot_manifold_canvas(patch[0:100], 10, type=plot_type, name="patch", make_dir="patch_img")

        s_train_holder = []
        g_train_holder = []
        s_test_holder = []
        g_test_holder = []
        for j in range(int(100//conf.batch_size)+1):
            feed_dict_1 = {X: train_data[conf.batch_size*j : conf.batch_size*j + conf.batch_size],
                           X_noised: train_data[conf.batch_size*j : conf.batch_size*j + conf.batch_size], phase: False}
            feed_dict_2 = {X_noised: test_data[conf.batch_size*j : conf.batch_size*j + conf.batch_size], phase: False}
            s_train_img, g_train_img = sess.run([s_out, g_out], feed_dict = feed_dict_1)
            s_test_img, g_test_img = sess.run([s_out, g_out], feed_dict=feed_dict_2)
            s_train_holder.append(s_train_img)
            g_train_holder.append(g_train_img)
            s_test_holder.append(s_test_img)
            g_test_holder.append(g_test_img)
        s_train_holder = np.reshape(s_train_holder, [-1, H, W, C])
        g_train_holder = np.reshape(g_train_holder, [-1, H, W, C])
        s_test_holder = np.reshape(s_test_holder,  [-1, H, W, C])
        g_test_holder = np.reshape(g_test_holder,  [-1, H, W, C])

        plot_manifold_canvas(s_train_holder[0:conf.n_img_plot], int(conf.n_img_plot**0.5), type = plot_type, name = s_train_name, make_dir = "s_img_train")
        plot_manifold_canvas(g_train_holder[0:conf.n_img_plot], int(conf.n_img_plot**0.5), type = plot_type, name = g_train_name, make_dir = "g_img_train")
        plot_manifold_canvas(s_test_holder[0:conf.n_img_plot], int(conf.n_img_plot**0.5), type = plot_type, name = s_test_name, make_dir = "s_img_test")
        plot_manifold_canvas(g_test_holder[0:conf.n_img_plot], int(conf.n_img_plot**0.5), type = plot_type, name = g_test_name, make_dir = "g_img_test")

        if conf.n_z == 2:
            x_vae = []
            y_vae = []
            label = []
            x_ae = []
            y_ae = []
            for k in range(total_batch_test):
                batch_xs, batch_noised_xs, batch_ys = FLAG_1_Data.next_batch(test_data, test_data, test_labels, conf.batch_size)
                feed_dict = {X: batch_xs, X_noised: batch_xs, phase: False}
                z_vae, z_ae = sess.run([s_latent, g_latent], feed_dict=feed_dict)
                for l in range(conf.batch_size):
                    x_vae.append(z_vae[l][0])
                    y_vae.append(z_vae[l][1])
                    x_ae.append(z_ae[l][0])
                    y_ae.append(z_ae[l][1])
                    label.append(batch_ys[l])

            plot_2d_scatter(x_vae, y_vae, label, name="VAE_latent_scatter_" + str(i))
            plot_2d_scatter(x_ae, y_ae, label, name="AE_latent_scatter_" + str(i))

    hour = int((time.time() - start_time) / 3600)
    min = int(((time.time() - start_time) - 3600 * hour) / 60)
    sec = int((time.time() - start_time) - 3600 * hour - 60 * min)
    print("Epoch: %d    lr_PVAE: %f    lr_G_D: %f    D_loss: %f\nPVAE_loss: %f    G_loss: %f\nM_Value: %f    K_Value: %f\ntime: %d hour %d min %d sec\n"
          %(i, conf.lr*2, conf.lr, l_d, l_s, l_g, M_value, K_value, hour, min, sec))

sess.close()