from data_utils import *
from utils import *
from plot import *
from EBAnoGAN import *
import time
import pdb

DEFINE_string("data", "MNIST", "[MNIST|MNIST_Fashion|CIFAR|CALTECH]")

DEFINE_integer("n_epoch", 100, "number of Epoch for training")
DEFINE_integer("n_z", 128, "Dimension of Latent Variables")
DEFINE_integer("batch_size", 128, "Batch Size for training")
DEFINE_integer("margin", 10, "Margin for D_loss")

DEFINE_float("lr", 0.0005, "learning rate for training")
DEFINE_float("KL_rate", 0.2, "Weight for Regularization loss")
DEFINE_float("PL_rate", 1.0, "Weight for Perceptual loss")
DEFINE_float("E_rate", 0.2, "Weight for Potential loss")
DEFINE_float("pt_rate", 0.1, "Weight for Pullaway Term")
DEFINE_float("FL_rate", 1.0, "Weight for Fake loss")

conf = print_user_flags(line_limit = 100)
print("-"*80)

if conf.data == "MNIST" or "MNIST_Fashion":
    plot_type = "C_1"
else:
    plot_type = "C_3"

FLAG_1_Data = data_controller(type = "MNIST",
                              n_channel = 1,
                              normal = [0,1,2,3,4],
                              anomalus = [5,6,7,8,9],
                              num_normal_train = 1000,
                              num_normal_test = 4500,
                              num_abnormal_test = 500,
                              name = "FLAG_1")

train_data,test_data = FLAG_1_Data.preprocessing()
print("performing preprocessing, augmentatioin...")
noised_train_data = salt_pepper_noise(train_data)
print("")
plot_manifold_canvas(train_data[0:100],10, type = plot_type, name ="original_images")
plot_manifold_canvas(noised_train_data[0:100],10, type = plot_type, name ="noised_images")
plot_manifold_canvas(test_data[0:100],10, type = plot_type, name = "test_images")

shape = np.shape(train_data)
W = shape[1]
H = shape[2]
C = shape[3]

X = tf.placeholder(tf.float32, shape = [conf.batch_size, W,H,C], name = "inputs")
X_noised = tf.placeholder(tf.float32, shape = [conf.batch_size, W,H,C], name = "inputs_noised")
phase = tf.placeholder(tf.bool, name = "training_phase")
lr = tf.placeholder(tf.float32, name = "lr")
global_step = tf.Variable(0, trainable = False, name = "global_step")

EBAnoGAN = EBAnoGAN(conf,shape, [128,256,512,1024])
X_out, D_loss, G_loss_VAE, G_loss_GAN, D_real, D_fake = EBAnoGAN.Revised_EBGAN(X,X_noised,phase)

total_batch = FLAG_1_Data.get_total_batch(train_data, conf.batch_size)
FLAG_1_Data.initialize_batch()

total_vars = tf.trainable_variables()

G_e_vars = [var for var in total_vars if "gaussian_encoder" in var.name]
G_d_vars = [var for var in total_vars if "bernoulli_decoder" in var.name]
G_vars = G_e_vars + G_d_vars

D_e_vars = [var for var in total_vars if "EB_encoder" in var.name]
D_d_vars = [var for var in total_vars if "EB_decoder" in var.name]
D_vars = D_e_vars + D_d_vars

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    op_D = tf.train.AdamOptimizer(learning_rate = lr/5).minimize(D_loss, var_list = D_vars, global_step = global_step)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    op_G_VAE = tf.train.AdamOptimizer(learning_rate = lr).minimize(G_loss_VAE, var_list = G_vars, global_step = global_step)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    op_G_GAN = tf.train.AdamOptimizer(learning_rate = lr).minimize(G_loss_GAN, var_list = G_vars, global_step = global_step)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

start_time = time.time()
for i in range(conf.n_epoch):
    d_loss = 0
    g_loss_vae = 0
    g_loss_gan = 0
    if i >= int(conf.n_epoch/2):
        lr_ = conf.lr/5
    else:
        lr_ = conf.lr
    for j in range(total_batch):
        batch_xs, batch_noised_xs = FLAG_1_Data.next_batch(train_data, noised_train_data, conf.batch_size)
        feed_dict= {X: batch_xs, X_noised: batch_noised_xs, phase: True, lr: lr_}
        d_,o_d, D_r, D_f, g = sess.run([D_loss, op_D, D_real, D_fake, global_step], feed_dict = feed_dict)
        g_vae, o_g_vae, g = sess.run([G_loss_VAE, op_G_VAE, global_step], feed_dict = feed_dict)
        g_gan, o_g_gan, g = sess.run([G_loss_GAN, op_G_GAN, global_step], feed_dict=feed_dict)
        d_loss +=d_/total_batch
        g_loss_vae +=g_vae/total_batch
        g_loss_gan +=g_gan/total_batch


    if i % 1 == 0 or i ==(conf.n_epoch -1):
        d_real_name = "dis_real_canvas" + str(i)
        d_fake_name = "dis_fake_canvas" + str(i)
        plot_manifold_canvas(D_r[0:100], 10, type = plot_type, name = d_real_name)
        plot_manifold_canvas(D_f[0:100], 10, type = plot_type, name = d_fake_name)

    if i % 1 == 0 or i == (conf.n_epoch -1):
        images_plot = sess.run(X_out, feed_dict = {X_noised: test_data[0:128], phase: False})
        name = "Manifold_canvas_" + str(i)
        plot_manifold_canvas(images_plot[0:100], 10, type = plot_type, name = name)

    hour = int((time.time() - start_time) / 3600)
    min = int(((time.time() - start_time) - 3600 * hour) / 60)
    sec = int((time.time() - start_time) - 3600 * hour - 60 * min)
    print("Epoch: %d    lr_D: %f    lr_G: %f    D_loss: %f    G_loss_vae: %f    G_loss_GAN: %f    \ntime: %d hour %d min %d sec\n"
          %(i, lr_/5, lr_, d_loss, g_loss_vae, g_loss_gan, hour, min, sec))