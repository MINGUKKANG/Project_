from data_utils import *
from utils import *
from plot import *
from EBAnoGAN import *
import time
import pdb

DEFINE_string("data", "MNIST", "[MNIST|MNIST_Fashion|CIFAR|CALTECH]")

DEFINE_integer("n_epoch", 50, "number of Epoch for training")
DEFINE_integer("n_z", 128, "Dimension of Latent Variables")
DEFINE_integer("batch_size", 64, "Batch Size for training")
DEFINE_integer("margin", 1, "Margin for D_loss")
DEFINE_integer("n_img_plot", 64, "Number of Images for plotting")

DEFINE_float("lr", 0.0005, "learning rate for training")
DEFINE_float("KL_rate", 0.5, "Weight for Regularization loss")
DEFINE_float("PVAE_rate", 1/3, "Weight for ori2fake loss")
DEFINE_float("AE_rate", 1/3, "Weight for fake2ori loss")
DEFINE_float("PT_rate", 0.1, "Weight for Pullaway Term")
DEFINE_float("DIS_rate", 1/3, "Weight for Fake loss")
DEFINE_float("beta1", 0.5, "")
conf = print_user_flags(line_limit = 100)
print("-"*80)

if conf.data == "MNIST" or "MNIST_Fashion":
    plot_type = "C_1"
else:
    plot_type = "C_3"

FLAG_1_Data = data_controller(type = "MNIST",
                              n_channel = 1,
                              enlr_size = 64,
                              normal = [0,1,2,3,4],
                              anomalus = [5,6,7,8,9],
                              num_normal_train = 20000,
                              num_normal_test = 4500,
                              num_abnormal_test = 500,
                              name = "FLAG_1")

train_data, train_labels, test_data, test_labels = FLAG_1_Data.preprocessing()
print("performing preprocessing, augmentatioin...")
noised_train_data = salt_pepper_noise(train_data)
print("")

plot_manifold_canvas(train_data[0:conf.n_img_plot],int(conf.n_img_plot**0.5), type = plot_type, make_dir = "f_img_train", name ="ori_img_train")
plot_manifold_canvas(noised_train_data[0:conf.n_img_plot],int(conf.n_img_plot**0.5), type = plot_type, make_dir = "f_img_train", name ="noised_img_train")
plot_manifold_canvas(test_data[0:conf.n_img_plot],int(conf.n_img_plot**0.5), type = plot_type, make_dir = "f_img_test", name = "ori_img_test")
plot_manifold_canvas(train_data[0:conf.n_img_plot],int(conf.n_img_plot**0.5), type = plot_type, make_dir = "r_img_train", name ="ori_img_train")
plot_manifold_canvas(noised_train_data[0:conf.n_img_plot],int(conf.n_img_plot**0.5), type = plot_type, make_dir = "r_img_train", name ="noised_img_train")
plot_manifold_canvas(test_data[0:conf.n_img_plot],int(conf.n_img_plot**0.5), type = plot_type, make_dir = "r_img_test", name = "ori_img_test")

shape = np.shape(train_data)
W = shape[1]
H = shape[2]
C = shape[3]

X = tf.placeholder(tf.float32, shape = [conf.batch_size, W,H,C], name = "inputs")
X_noised = tf.placeholder(tf.float32, shape = [conf.batch_size, W,H,C], name = "inputs_noised")
phase = tf.placeholder(tf.bool, name = "training_phase")
lr = tf.placeholder(tf.float32, name = "lr")

ADGAN = ADGAN(conf, shape, [128,256,512,1024])
z_r2f, f_out, ori2fake_loss, z_f2r, r_out, fake2ori_loss, PT_loss, real_loss, G_loss, D_loss = ADGAN.ADGAN(X,X_noised,phase)

total_batch_train = FLAG_1_Data.get_total_batch(train_data, conf.batch_size)
total_batch_test = FLAG_1_Data.get_total_batch(test_data, conf.batch_size)
FLAG_1_Data.initialize_batch()

total_vars = tf.trainable_variables()

r2f_e_vars = [var for var in total_vars if "gaussian_encoder" in var.name]
r2f_d_vars = [var for var in total_vars if "bernoulli_decoder" in var.name]
r2f_vars = r2f_e_vars + r2f_d_vars

f2r_e_vars = [var for var in total_vars if "Cycle_encoder" in var.name]
f2r_d_vars = [var for var in total_vars if "Cycle_decoder" in var.name]
f2r_vars = f2r_e_vars + f2r_d_vars

D_e_vars = [var for var in total_vars if "EB_encoder" in var.name]
D_d_vars = [var for var in total_vars if "EB_decoder" in var.name]
D_vars = D_e_vars + D_d_vars

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    op_D = tf.train.AdamOptimizer(learning_rate = lr, beta1 = conf.beta1).minimize(D_loss, var_list = D_vars)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    op_G = tf.train.AdamOptimizer(learning_rate = lr/2, beta1 = conf.beta1).minimize(G_loss, var_list = r2f_vars + f2r_vars)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

start_time = time.time()
for i in range(conf.n_epoch):
    l_d = 0
    l_rf = 0
    l_fr = 0
    l_pt = 0
    l_g = 0
    for j in range(total_batch_train):
        batch_xs, batch_noised_xs, batch_ys = FLAG_1_Data.next_batch(train_data, noised_train_data,train_labels ,conf.batch_size)
        feed_dict= {X: batch_xs, X_noised: batch_noised_xs, phase: True, lr: conf.lr}
        _, d = sess.run([op_D, D_loss], feed_dict = feed_dict)
        _, rf, fr, pt, g = sess.run([op_G, ori2fake_loss, fake2ori_loss, PT_loss, G_loss], feed_dict = feed_dict)
        l_d += d/total_batch_train
        l_rf += rf/total_batch_train
        l_fr += fr/total_batch_train
        l_pt += pt/total_batch_train
        l_g += g/total_batch_train

    if i % 1 == 0 or i ==(conf.n_epoch -1):
        f_train_name = "f_img_train_" + str(i)
        f_test_name = "f_img_test_" + str(i)
        r_train_name = "r_img_train" + str(i)
        r_test_name = "r_img_test" + str(i)
        feed_dict_1 = {X_noised: train_data[0:conf.batch_size], phase: False}
        feed_dict_2 = {X_noised: test_data[0:conf.batch_size], phase: False}
        f_train_img, r_train_img  = sess.run([f_out, r_out], feed_dict = feed_dict_1)
        f_test_img, r_test_img = sess.run([f_out, r_out], feed_dict=feed_dict_2)
        plot_manifold_canvas(f_train_img[0:conf.n_img_plot], int(conf.n_img_plot**0.5), type = plot_type, name = f_train_name, make_dir = "f_img_train")
        plot_manifold_canvas(r_train_img[0:conf.n_img_plot], int(conf.n_img_plot**0.5), type = plot_type, name = r_train_name, make_dir = "r_img_train")
        plot_manifold_canvas(f_test_img[0:conf.n_img_plot], int(conf.n_img_plot**0.5), type = plot_type, name = f_test_name, make_dir = "f_img_test")
        plot_manifold_canvas(r_test_img[0:conf.n_img_plot], int(conf.n_img_plot**0.5), type = plot_type, name = r_test_name, make_dir = "r_img_test")

        if conf.n_z == 2:
            x_vae = []
            y_vae = []
            label = []
            x_ae = []
            y_ae = []
            for k in range(total_batch_test):
                batch_xs, batch_noised_xs, batch_ys = FLAG_1_Data.next_batch(test_data, test_data, test_labels,
                                                                             conf.batch_size)
                feed_dict = {X: batch_xs, X_noised: batch_xs, phase: False}
                z_vae, z_ae = sess.run([z_r2f, z_f2r], feed_dict=feed_dict)
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
    print("Epoch: %d    lr_D: %f    lr_G: %f    D_loss: %f\nPVAE_loss: %f    AE_loss: %f    PT_loss: %f    G_loss: %f\ntime: %d hour %d min %d sec\n"
          %(i, conf.lr, conf.lr/2, l_d, l_rf, l_fr, l_pt, l_g, hour, min, sec))

sess.close()