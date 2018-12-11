# Project_
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
