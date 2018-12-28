import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from data_utils import *

def plot_2d_scatter(x,y,test_labels, name):

    plt.figure(figsize = (8,6))
    plt.scatter(x,y, c = test_labels, marker ='.', edgecolor = 'none', cmap = discrete_cmap('jet'))
    plt.colorbar()
    plt.grid()
    if not tf.gfile.Exists("./plot/Scatter"):
        tf.gfile.MakeDirs("./plot/Scatter")
    name = name + ".png"
    plt.savefig('./plot/Scatter/' + name)
    plt.close()

def discrete_cmap(base_cmap =None):

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0,1,10))
    cmap_name = base.name + str(10)

    return base.from_list(cmap_name,color_list,10)

def plot_manifold_canvas(images, n, type, name, make_dir = None):

    images = cv2.normalize(images, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    assert images.shape[0] == n**2, "n**2 should be number of images"
    height = images.shape[1]
    width = images.shape[2] # width = height
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)

    if type == "C_1":
        canvas = np.empty((n * height, n * height))
        for i, yi in enumerate(x):
            for j, xi in enumerate(y):
                canvas[height*i: height*i + height, width*j: width*j + width] = np.reshape(images[n*i + j], [height, width])
        plt.figure(figsize=(8, 8))
        plt.imshow(canvas, cmap="gray")
    else:
        canvas = np.empty((n * height, n * height, 3))
        for i, yi in enumerate(x):
            for j, xi in enumerate(y):
                canvas[height*i: height*i + height, width*j: width*j + width,:] = images[n*i + j]
        plt.figure(figsize=(8, 8))
        plt.imshow(canvas)

    path = "./plot"
    if not tf.gfile.Exists("./plot"):
        tf.gfile.MakeDirs("./plot")
    if make_dir is not None:
        if not tf.gfile.Exists("./plot/" + str(make_dir)):
            tf.gfile.MakeDirs("./plot/" + str(make_dir))
        path = "./plot/" + str(make_dir)

    name = name + ".png"
    path = os.path.join(path, name)
    plt.savefig(path)
    print("saving location: %s" % (path))
    plt.close()

def plot_roc_curve(fpr, tpr, name, make_dir = None):
    plt.plot(fpr, tpr,'r', label = 'Bend GAN') # Boundary Equilibrium Novelty Detection GAN
    plt.xlabel('False Positive Rate(FPR)')
    plt.ylabel('True Positive Rate(TPR)')
    plt.title('Receiver operating Characteristic(ROC Curve)')
    plt.legend(loc = 'upper right')

    path = "./plot"
    if make_dir is not None:
        path = "./plot/" + str(make_dir)
        if not tf.gfile.Exists(path):
            tf.gfile.MakeDirs(path)
    name = name + ".png"
    path = os.path.join(path,name)
    plt.savefig(path)
    print("saving location: %s" %(path))
    plt.close()