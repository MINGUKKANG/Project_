from data_utils import *
from utils import *
from plot import *
from EBAnoGAN import *
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import time
import cv2
import pdb

# Boundary Equilibrium Novelty Detection GAN

DEFINE_string("mode", "MNIST", "[MNIST|MNIST_Fashion|CIFAR10|CALTECH|enter dataset name]")
DEFINE_integer("channel", 1, "Number of channels for you own dataset")

DEFINE_boolean("PVAE", True, "If False, just apply VAE instead of PAVE")
DEFINE_boolean("make_patches", False, "boolean for patches GAN")
DEFINE_boolean("pretrain_pvae", True, "boolean for pretraining PVAE")
DEFINE_integer("patch_size", 32, "patch_size for EB Discriminator")
DEFINE_integer("patch_strides", 4, "patch strides for EB Discriminator")
DEFINE_integer("pre_epoch", 10, "epochs for pretraining of PVAE")
DEFINE_integer("n_epoch", 80, "number of Epoch for training")
DEFINE_integer("n_z", 128, "Dimension of Latent Variables")
DEFINE_integer("batch_size", 64, "Batch Size for training")
DEFINE_integer("n_img_plot", 100, "Number of Images for plotting")
DEFINE_integer("decay_epoch", 80, "interval of epoch for learning rate decay")

DEFINE_float("lr_PVAE", 0.0005, "learning rate for PVAE training")
DEFINE_float("lr", 0.0008, "learning rate for GAN training")
DEFINE_float("PVAE_rate", 1.0, "Weight for ori2fake loss")
DEFINE_float("beta1", 0.5, "")
DEFINE_float("KL_rate", 0.2, "Weight for Regularization loss")
DEFINE_float("Cycle_lamda", 1.0, "Weight for Cycle loss")
DEFINE_float("gamma", 0.6, "Hyper parameter of BEGAN")
DEFINE_float("lamda", 0.002, "Hyper parameter of BEGAN")

conf = print_user_flags(line_limit = 100)
print("-"*80)

print("performing preprocessing...")
data_controller = data_controller()
if conf.mode == "MNIST":
    train_data, train_labels, test_data, test_labels = data_controller.call_benchmark_data(mode = conf.mode,
                                                                                       normal = [6],
                                                                                       abnormal = [0,1,2,3,4,5,7,8,9])
else:
    train_data, train_labels, test_data, test_labels = data_controller.call_own_dataset(mode = conf.mode,
                                                                                        channel = conf.channel,
                                                                                        enlr_size = 128,
                                                                                        normal = [0],
                                                                                        abnormal = [1],
                                                                                        n_train = 3000,
                                                                                        an_train = 0,
                                                                                        n_test = 1000,
                                                                                        an_test = 50)
shape = np.shape(train_data)
H = shape[1]
W = shape[2]
C = shape[3]

if C == 1:
    plot_type = "C_1"
else:
    plot_type = "C_3"

noised_train_data = salt_pepper_noise(train_data)
print("")

plot_manifold_canvas(train_data[0:conf.n_img_plot],int(conf.n_img_plot**0.5), type = plot_type, make_dir = "s_img_train", name ="ori_img_train")
plot_manifold_canvas(noised_train_data[0:conf.n_img_plot],int(conf.n_img_plot**0.5), type = plot_type, make_dir = "s_img_train", name ="noised_img_train")
plot_manifold_canvas(test_data[0:conf.n_img_plot],int(conf.n_img_plot**0.5), type = plot_type, make_dir = "s_img_test", name = "ori_img_test")
plot_manifold_canvas(train_data[0:conf.n_img_plot],int(conf.n_img_plot**0.5), type = plot_type, make_dir = "g_img_train", name ="ori_img_train")
plot_manifold_canvas(noised_train_data[0:conf.n_img_plot],int(conf.n_img_plot**0.5), type = plot_type, make_dir = "g_img_train", name ="noised_img_train")
plot_manifold_canvas(test_data[0:conf.n_img_plot],int(conf.n_img_plot**0.5), type = plot_type, make_dir = "g_img_test", name = "ori_img_test")

X = tf.placeholder(tf.float32, shape = [conf.batch_size, W,H,C], name = "inputs")
X_noised = tf.placeholder(tf.float32, shape = [conf.batch_size, W,H,C], name = "inputs_noised")
phase = tf.placeholder(tf.bool, name = "training_phase")
lr_pvae = tf.placeholder(tf.float32 ,name = "learning")

ADGAN = ADGAN(conf, shape, [128,256,512,1024])
s_latent, s_out, S_loss, g_latent, g_out, X_patches, D_loss, G_loss, Heat_map, D_value, M, K, update_k = ADGAN.ADGAN(X,X_noised,phase)

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
    op_S = tf.train.AdamOptimizer(learning_rate = lr_pvae, beta1 = conf.beta1).minimize(S_loss, var_list = S_vars)
'''
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    op_C = tf.train.AdamOptimizer(learning_rate = conf.lr, beta1 = conf.beta1).minimize(C_loss, var_list = G_vars)
'''
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    op_G = tf.train.AdamOptimizer(learning_rate = conf.lr, beta1 = conf.beta1).minimize(G_loss, var_list = G_vars)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    op_D = tf.train.RMSPropOptimizer(learning_rate = conf.lr).minimize(D_loss, var_list = D_vars)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

tower = batch_tower(train_data, train_labels, test_data, test_labels, conf.batch_size)
total_batch_train, total_batch_test = tower.get_total_batch()
start_time = time.time()
m_holder = []
k_holder = []

if conf.pretrain_pvae is True:
    for i in range(conf.pre_epoch):
        for j in range(total_batch_train):
            batch_xs, batch_ys = tower.train_nb()
            batch_noised_xs = salt_pepper_noise(batch_xs)
            feed_dict = {X: batch_xs, X_noised: batch_noised_xs, phase: True, lr_pvae: conf.lr_PVAE}
            _ = sess.run(op_S, feed_dict = feed_dict)

for i in range(conf.n_epoch):
    lr_pvae_decay = conf.lr_PVAE/(2**int(i/conf.decay_epoch))
    l_s = 0
    # l_c = 0
    l_d = 0
    l_g = 0
    for j in range(total_batch_train):
        batch_xs, batch_ys = tower.train_nb()
        batch_noised_xs = salt_pepper_noise(batch_xs)
        feed_dict= {X: batch_xs, X_noised: batch_noised_xs, phase: True, lr_pvae: lr_pvae_decay}
        _, s = sess.run([op_S, S_loss], feed_dict = feed_dict)
        # _, c = sess.run([op_C, C_loss], feed_dict = feed_dict)
        _, d = sess.run([op_D, D_loss], feed_dict = feed_dict)
        _, g = sess.run([op_G, G_loss], feed_dict = feed_dict)
        _, M_value, K_value = sess.run([update_k, M, K], feed_dict =feed_dict)
        l_s += s/total_batch_train
        # l_c += c/total_batch_train
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
        HM_holder = []
        HM_holder_1 = []
        test_data_1 = []

        for j in range(int(100//conf.batch_size)+1):
            feed_dict_1 = {X: train_data[conf.batch_size*j : conf.batch_size*j + conf.batch_size],
                           X_noised: train_data[conf.batch_size*j : conf.batch_size*j + conf.batch_size], phase: False}
            feed_dict_2 = {X: test_data[conf.batch_size*j : conf.batch_size*j + conf.batch_size],
                           X_noised: test_data[conf.batch_size*j : conf.batch_size*j + conf.batch_size], phase: False}
            s_train_img, g_train_img = sess.run([s_out, g_out], feed_dict = feed_dict_1)
            s_test_img, g_test_img, HM, = sess.run([s_out, g_out, Heat_map], feed_dict=feed_dict_2)
            s_train_holder.append(s_train_img)
            g_train_holder.append(g_train_img)
            s_test_holder.append(s_test_img)
            g_test_holder.append(g_test_img)
            HM_holder.append(HM)
        s_train_holder = np.reshape(s_train_holder, [-1, H, W, C])
        g_train_holder = np.reshape(g_train_holder, [-1, H, W, C])
        s_test_holder = np.reshape(s_test_holder,  [-1, H, W, C])
        g_test_holder = np.reshape(g_test_holder,  [-1, H, W, C])
        HM_holder = np.reshape(HM_holder, [-1, H, W, C])
        test_data = test_data.astype(np.float32)
        if C == 1:
            for m in range(len(HM_holder)):
                test_data_1.append(cv2.normalize(cv2.cvtColor(test_data[m], cv2.COLOR_GRAY2RGB),
                                                 None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC3))
                HM_holder_1.append(cv2.normalize(cv2.cvtColor(HM_holder[m], cv2.COLOR_GRAY2RGB),
                                                 None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC3))
                HM_holder_1[m] = cv2.cvtColor(cv2.applyColorMap(HM_holder_1[m], cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                HM_holder_1[m] = cv2.addWeighted(HM_holder_1[m], 0.5, test_data_1[m], 0.5, 0)
        else:
            for n in range(len(HM_holder)):
                test_data_1.append(cv2.normalize(test_data[n], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC3))
                HM_holder_1.append(cv2.normalize(HM_holder[n], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC3))
                HM_holder_1[n] = cv2.cvtColor(cv2.applyColorMap(HM_holder_1[n], cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                HM_holder_1[n] = cv2.addWeighted(HM_holder_1[n], 0.5, test_data_1[n], 0.5, 0)

        plot_manifold_canvas(s_train_holder[0:conf.n_img_plot], int(conf.n_img_plot**0.5), type = plot_type, name = s_train_name, make_dir = "s_img_train")
        plot_manifold_canvas(g_train_holder[0:conf.n_img_plot], int(conf.n_img_plot**0.5), type = plot_type, name = g_train_name, make_dir = "g_img_train")
        plot_manifold_canvas(s_test_holder[0:conf.n_img_plot], int(conf.n_img_plot**0.5), type = plot_type, name = s_test_name, make_dir = "s_img_test")
        plot_manifold_canvas(g_test_holder[0:conf.n_img_plot], int(conf.n_img_plot**0.5), type = plot_type, name = g_test_name, make_dir = "g_img_test")
        plot_manifold_canvas(np.asarray(HM_holder_1[0:conf.n_img_plot]), int(conf.n_img_plot**0.5), type = "C_3", name = "HeatMaps", make_dir = "Heatmaps")

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

    D_holder = []
    for kk in range(total_batch_test):
        batch_xs, batch_ys = tower.test_nb()
        batch_noised_xs = batch_xs
        feed_dict = {X: batch_xs, X_noised: batch_xs, phase: False}
        D_v = sess.run(D_value, feed_dict = feed_dict)
        for jj in range(len(D_v)):
            D_holder.append(D_v[jj])

    A = (total_batch_test - 1)*conf.batch_size
    B = len(test_labels) - (total_batch_test - 1)*conf.batch_size
    fpr, tpr, thresholds = roc_curve(test_labels, np.concatenate((D_holder[0:A], D_holder[-1*B:]), axis =0))
    plot_roc_curve(fpr = fpr, tpr = tpr, name = "ROC_Curve", make_dir = "ROC_Curve")
    auc_value = auc(fpr, tpr)

    hour = int((time.time() - start_time) / 3600)
    min = int(((time.time() - start_time) - 3600 * hour) / 60)
    sec = int((time.time() - start_time) - 3600 * hour - 60 * min)
    print("Epoch: %d    lr_VAE: %f    lr_G_D: %f\nVAE_loss: %f" %(i, lr_pvae_decay, conf.lr, l_s))
    print("D_loss: %f    G_loss: %f\nM_value: %f    K_value: %f    AUC: %f\ntime: %d hour %d min %d sec" %(l_d, l_g, M_value, K_value, auc_value, hour, min, sec))
    print("-"*80)

sess.close()