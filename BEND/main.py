from data_utils import *
from utils import *
from plot import *
from EBAnoGAN import *
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import time
import cv2
import pdb

import gzip
import pickle

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 1, 28, 28).astype(np.float32)
    return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

def extract_norm_and_out(X, y, normal, outlier):
    idx_normal = np.any(y[..., None] == np.array(normal)[None, ...], axis=1)
    idx_outlier = np.any(y[..., None] == np.array(outlier)[None, ...], axis=1)

    X_normal = X[idx_normal]
    y_normal = np.zeros(np.sum(idx_normal), dtype=np.uint8)

    X_outlier = X[idx_outlier]
    y_outlier = np.ones(np.sum(idx_outlier), dtype=np.uint8)

    return X_normal, X_outlier, y_normal, y_outlier

'''
data_path = "./data/"
X = load_mnist_images('%strain-images-idx3-ubyte.gz' % data_path)
y = load_mnist_labels('%strain-labels-idx1-ubyte.gz' % data_path)
X_test = load_mnist_images('%st10k-images-idx3-ubyte.gz' % data_path)
y_test = load_mnist_labels('%st10k-labels-idx1-ubyte.gz' % data_path)

X_norm, X_out, y_norm, y_out = extract_norm_and_out(X,y, normal = [0], outlier = [1,2,3,4,5,6,7,8,9])
n_norm = len(y_norm)
floatX = np.float32
out_frac = floatX(.1)
n_out = int(np.ceil(out_frac * n_norm / (1 - out_frac)))
np.random.seed(0)
perm_norm = np.random.permutation(len(y_norm))
perm_out = np.random.permutation(len(y_out))
mnist_val_frac = 1./6
n_norm_split = int(mnist_val_frac * n_norm)
n_out_split = int(mnist_val_frac * n_out)
_X_train = np.concatenate((X_norm[perm_norm[n_norm_split:]], X_out[perm_out[:n_out][n_out_split:]]))
_y_train = np.append(y_norm[perm_norm[n_norm_split:]], y_out[perm_out[:n_out][n_out_split:]])
_X_val = np.concatenate((X_norm[perm_norm[:n_norm_split]], X_out[perm_out[:n_out][:n_out_split]]))
_y_val = np.append(y_norm[perm_norm[:n_norm_split]], y_out[perm_out[:n_out][:n_out_split]])

n_train = len(_y_train)
n_val = len(_y_val)
perm_train = np.random.permutation(n_train)
perm_val = np.random.permutation(n_val)
_X_train = _X_train[perm_train]
_y_train = _y_train[perm_train]
_X_val = _X_train[perm_val]
_y_val = _y_train[perm_val]

X_norm, X_out, y_norm, y_out = extract_norm_and_out(X_test, y_test, normal=[0], outlier=[1,2,3,4,5,6,7,8,9])
_X_test = np.concatenate((X_norm, X_out))
_y_test = np.append(y_norm, y_out)
perm_test = np.random.permutation(len(_y_test))
_X_test = _X_test[perm_test]
_y_test = _y_test[perm_test]
n_test = len(_y_test)
#######
'''

data_path = "./data"
X, y = [], []
count = 1
filename = '%s/data_batch_%i' % (data_path, count)

while os.path.exists(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding = "bytes")
    X.append(batch[b'data'])
    y.append(batch[b'labels'])
    count += 1
    filename = '%s/data_batch_%i' % (data_path, count)

X = np.concatenate(X).reshape(-1, 3, 32, 32).astype(np.float32)
y = np.concatenate(y).astype(np.int32)

path = '%s/test_batch' % data_path
with open(path, 'rb') as f:
    batch = pickle.load(f, encoding = 'bytes')

X_test = batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32)
y_test = np.array(batch[b'labels'], dtype=np.int32)

normal = [0]
outliers = [1,2,3,4,5,6,7,8,9]
floatX = np.float32
out_frac = floatX(.1)
X_norm, X_out, y_norm, y_out = extract_norm_and_out(X, y, normal=normal, outlier=outliers)
n_norm = len(y_norm)
n_out = int(np.ceil(out_frac * n_norm / (1 - out_frac)))
np.random.seed(0)
perm_norm = np.random.permutation(len(y_norm))
perm_out = np.random.permutation(len(y_out))

cifar10_val_frac = 1./5
n_norm_split = int(cifar10_val_frac * n_norm)
n_out_split = int(cifar10_val_frac * n_out)

_X_train = np.concatenate((X_norm[perm_norm[n_norm_split:]], X_out[perm_out[:n_out][n_out_split:]]))
_y_train = np.append(y_norm[perm_norm[n_norm_split:]], y_out[perm_out[:n_out][n_out_split:]])
_X_val = np.concatenate((X_norm[perm_norm[:n_norm_split]], X_out[perm_out[:n_out][:n_out_split]]))
_y_val = np.append(y_norm[perm_norm[:n_norm_split]], y_out[perm_out[:n_out][:n_out_split]])

n_train = len(_y_train)
n_val = len(_y_val)
perm_train = np.random.permutation(n_train)
perm_val = np.random.permutation(n_val)
_X_train = _X_train[perm_train]
_y_train = _y_train[perm_train]
_X_val = _X_train[perm_val]
_y_val = _y_train[perm_val]

# test set
X_norm, X_out, y_norm, y_out = extract_norm_and_out(X_test, y_test, normal=normal, outlier=outliers)
_X_test = np.concatenate((X_norm, X_out))
_y_test = np.append(y_norm, y_out)
perm_test = np.random.permutation(len(_y_test))
_X_test = _X_test[perm_test]
_y_test = _y_test[perm_test]
n_test = len(_y_test)

# Boundary Equilibrium Novelty Detection GAN

DEFINE_string("data", "CIFAR", "[MNIST|MNIST_Fashion|CIFAR|CALTECH]")
DEFINE_integer("n_channel", 3, "Number of channels")

DEFINE_boolean("PVAE", True, "If False, just apply VAE instead of PAVE")
DEFINE_boolean("make_patches", False, "boolean for patches GAN")
DEFINE_boolean("pretrain_pvae", False, "boolean for pretraining PVAE")
DEFINE_integer("n_epoch", 120, "number of Epoch for training")
DEFINE_integer("n_z", 128, "Dimension of Latent Variables")
DEFINE_integer("batch_size", 64, "Batch Size for training")
DEFINE_integer("pre_epoch", 5, "epochs for pretraining of PVAE")
DEFINE_integer("n_img_plot", 100, "Number of Images for plotting")
DEFINE_integer("patch_size", 32, "patch_size for EB Discriminator")
DEFINE_integer("patch_strides", 4, "patch strides for EB Discriminator")
DEFINE_integer("decay_epoch", 40, "interval of epoch for learning rate decay")

DEFINE_float("lr_PVAE", 0.0008, "learning rate for PVAE training")
DEFINE_float("lr", 0.0005, "learning rate for GAN training")
DEFINE_float("PVAE_rate", 1.0, "Weight for ori2fake loss")
DEFINE_float("beta1", 0.5, "")

# Hyper parameter optimized by Bayesian Optimization
DEFINE_float("KL_rate", 0.5, "Weight for Regularization loss")
DEFINE_float("Cycle_lamda", 1.0, "Weight for Cycle loss")
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
                              enlr_size = 32,
                              normal = [1],
                              anomalus = [0,2,3,4,5,6,7,8,9],
                              num_normal_train = 4000,
                              num_normal_test = 1000,
                              num_abnormal_test = 9000,
                              name = "FLAG_1")

train_data, train_labels, test_data, test_labels, test_bi_labels = FLAG_1_Data.preprocessing()
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
lr_pvae = tf.placeholder(tf.float32 ,name = "learning")

ADGAN = ADGAN(conf, shape, [128,256,512,1024])
s_latent, s_out, S_loss, g_latent, g_out, X_patches, D_loss, G_loss, Heat_map, D_value, M, K, update_k = ADGAN.ADGAN(X,X_noised,phase)
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

start_time = time.time()
m_holder = []
k_holder = []

if conf.pretrain_pvae is True:
    for i in range(conf.pre_epoch):
        for j in range(total_batch_train):
            batch_xs, batch_noised_xs, batch_ys = FLAG_1_Data.next_batch(train_data, noised_train_data, train_labels, conf.batch_size)
            feed_dict = {X: batch_xs, X_noised: batch_noised_xs, phase: True, lr_pvae: conf.lr_PVAE}
            _ = sess.run(op_S, feed_dict = feed_dict)
    FLAG_1_Data.initialize_batch()

for i in range(conf.n_epoch):
    lr_pvae_decay = conf.lr_PVAE/(2**int(i/conf.decay_epoch))
    l_s = 0
    # l_c = 0
    l_d = 0
    l_g = 0
    for j in range(total_batch_train):
        batch_xs, batch_noised_xs, batch_ys = FLAG_1_Data.next_batch(train_data, noised_train_data,train_labels ,conf.batch_size)
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

        if conf.n_channel == 1:
            cv_type = cv2.CV_8UC1
            for m in range(len(HM_holder)):
                gray = cv2.normalize(test_data[m], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype= cv_type)
                test_data_1.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
        else:
            cv_type = cv2.CV_8UC3
            for n in range(len(HM_holder)):
                color = cv2.normalize(test_data[n], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype= cv_type)
                test_data_1.append(color)

        for k in range(len(HM_holder)):
            HM_holder_1.append(cv2.normalize(HM_holder[k], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv_type))
            HM_holder_1[k] = cv2.applyColorMap(HM_holder_1[k], cv2.COLORMAP_JET)
            HM_holder_1[k] = cv2.addWeighted(HM_holder_1[k], 0.8, test_data_1[k], 0.2, 0)

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
        batch_xs, batch_noised_xs, batch_ys = FLAG_1_Data.next_batch(test_data, test_data, test_labels, conf.batch_size)
        feed_dict = {X: batch_xs, X_noised: batch_xs, phase: False}
        D_v = sess.run(D_value, feed_dict = feed_dict)
        for jj in range(len(D_v)):
            D_holder.append(D_v[jj])
    fpr, tpr, thresholds = roc_curve(test_bi_labels[0:len(D_holder)], D_holder)
    plot_roc_curve(fpr = fpr, tpr = tpr, name = "ROC_Curve", make_dir = "ROC_Curve")
    auc_value = auc(fpr, tpr)


    hour = int((time.time() - start_time) / 3600)
    min = int(((time.time() - start_time) - 3600 * hour) / 60)
    sec = int((time.time() - start_time) - 3600 * hour - 60 * min)
    print("Epoch: %d    lr_VAE: %f    lr_G_D: %f\nVAE_loss: %f" %(i, lr_pvae_decay, conf.lr, l_s))
    print("D_loss: %f    G_loss: %f\nM_value: %f    K_value: %f    AUC: %f\ntime: %d hour %d min %d sec" %(l_d, l_g, M_value, K_value, auc_value, hour, min, sec))
    print("-"*80)

sess.close()