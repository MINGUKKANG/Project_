import pdb
import cv2
import numpy as np
from glob import *
import pickle
import gzip
import os
import random

random.seed(0)
np.random.seed(0)

class data_controller:
    def __init__(self):
        self.self = self

    def extract_norm_and_out(self, X, y, normal, outlier):
        idx_normal = np.any(y[..., None] == np.array(normal)[None, ...], axis=1)
        idx_outlier = np.any(y[..., None] == np.array(outlier)[None, ...], axis=1)
        X_normal = X[idx_normal]
        y_normal = np.zeros(np.sum(idx_normal), dtype=np.uint8)
        X_outlier = X[idx_outlier]
        y_outlier = np.ones(np.sum(idx_outlier), dtype=np.uint8)

        return X_normal, X_outlier, y_normal, y_outlier

    def load_mnist_images(self, filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28).astype(np.float32)

        return data

    def load_mnist_labels(self, filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)

        return data

    def enlr_data(self, data, enlr_size):
        length = len(data)
        c = int(np.shape(data)[3])

        holder = []
        for i in range(length):
            img = cv2.resize(data[i], (enlr_size, enlr_size), interpolation=cv2.INTER_CUBIC)
            holder.append(img)

        holder = np.reshape(holder,[length, 32 , 32, c])

        return holder

    def preprocessing(self, train, test):

        training_mean = np.mean(train, axis=(0, 1, 2))
        testing_mean = np.mean(test, axis=(0, 1, 2))
        train = (train - 127.5) / 127.5
        test = (test - 127.5) / 127.5
        print("Train_Data_mean: ", training_mean)
        print("Test_Data_mean: ", testing_mean)

        return train, test

    def call_benchmark_data(self, mode, normal, abnormal):
        if mode =="CIFAR10":
            data_path = "./data/%s/" % (mode)
            X, y = [], []
            count = 1
            filename = '%s/data_batch_%i' % (data_path, count)

            while os.path.exists(filename):
                with open(filename, 'rb') as f:
                    batch = pickle.load(f, encoding="bytes")
                X.append(batch[b'data'])
                y.append(batch[b'labels'])
                count += 1
                filename = '%s/data_batch_%i' % (data_path, count)

            X = np.concatenate(X).reshape(-1, 3, 32, 32).astype(np.float32)
            X = np.transpose(X, [0,2,3,1])
            y = np.concatenate(y).astype(np.int32)

            path = '%s/test_batch' % data_path
            with open(path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')

            X_test = batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32)
            X_test = np.transpose(X_test, [0,2,3,1])
            y_test = np.array(batch[b'labels'], dtype=np.int32)

            floatX = np.float32
            out_frac = floatX(.1)
            X_norm, X_out, y_norm, y_out = self.extract_norm_and_out(X, y, normal= normal, outlier= abnormal)
            n_norm = len(y_norm)
            n_out = int(np.ceil(out_frac * n_norm / (1 - out_frac)))
            perm_norm = np.random.permutation(len(y_norm))
            perm_out = np.random.permutation(len(y_out))

            cifar10_val_frac = 1. / 5
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
            _X_val = _X_val[perm_val]
            _y_val = _y_val[perm_val]

            X_norm, X_out, y_norm, y_out = self.extract_norm_and_out(X_test, y_test, normal= normal, outlier= abnormal)
            _X_test = np.concatenate((X_norm, X_out))
            _y_test = np.append(y_norm, y_out)
            perm_test = np.random.permutation(len(_y_test))
            _X_test = _X_test[perm_test]
            _y_test = _y_test[perm_test]
            n_test = len(_y_test)
            _X_train, _X_test = self.preprocessing(_X_train, _X_test)
            print("mode: ", mode)
            print("Train images: ", np.shape(_X_train))
            print("Test images: ", np.shape(_X_test))

        else:
            data_path = "./data/%s/" %( mode )
            X = self.load_mnist_images('%strain-images-idx3-ubyte.gz' % data_path)
            y = self.load_mnist_labels('%strain-labels-idx1-ubyte.gz' % data_path)
            X_test = self.load_mnist_images('%st10k-images-idx3-ubyte.gz' % data_path)
            y_test = self.load_mnist_labels('%st10k-labels-idx1-ubyte.gz' % data_path)
            X = np.transpose(X, [0, 2, 3, 1])
            X_test = np.transpose(X_test, [0, 2, 3, 1])
            X = self.enlr_data(X, 32)
            X_test = self.enlr_data(X_test, 32)

            X_norm, X_out, y_norm, y_out = self.extract_norm_and_out(X, y, normal=normal, outlier= abnormal)
            n_norm = len(y_norm)
            floatX = np.float32
            out_frac = floatX(.1)
            n_out = int(np.ceil(out_frac * n_norm / (1 - out_frac)))
            perm_norm = np.random.permutation(len(y_norm))
            perm_out = np.random.permutation(len(y_out))
            mnist_val_frac = 1. / 6
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
            _X_val = _X_val[perm_val]
            _y_val = _y_val[perm_val]

            X_norm, X_out, y_norm, y_out = self.extract_norm_and_out(X_test, y_test, normal=normal, outlier= abnormal)
            _X_test = np.concatenate((X_norm, X_out))
            _y_test = np.append(y_norm, y_out)
            perm_test = np.random.permutation(len(_y_test))
            _X_test = _X_test[perm_test]
            _y_test = _y_test[perm_test]
            n_test = len(_y_test)
            _X_train, _X_test = self.preprocessing(_X_train, _X_test)
            print("mode: ", mode)
            print("Train images: ", np.shape(_X_train))
            print("Test images: ", np.shape(_X_test))

        return _X_train, _y_train, _X_test, _y_test

    def call_own_dataset(self, mode, channel, enlr_size, normal, abnormal, n_train, an_train, n_test, an_test):
        imgs_n_name = []
        imgs_train = []
        label_n = []
        imgs_ab_name = []
        imgs_test =[]
        label_ab = []
        for i in normal:
            imgs_n_name = np.append(imgs_n_name, glob(os.path.join("./data/" + mode, str(i) + "/*.*")))
            n_instance = len(glob(os.path.join("./data/" + mode, str(i) + "/*.*")))
            for k in range(n_instance):
                label_n = np.append(label_n, 0.0)

        for j in abnormal:
            imgs_ab_name = np.append(imgs_ab_name, glob(os.path.join("./data/" + mode, str(j) + "/*.*")))
            n_instance = len(glob(os.path.join("./data/" + mode, str(j) + "/*.*")))
            for k in range(n_instance):
                label_ab = np.append(label_ab, 1.0)

        _X_train = np.append(imgs_n_name[0:n_train], imgs_ab_name[0:an_train])
        _y_train = np.append(label_n[0:n_train], label_ab[0:an_train])
        _X_test = np.append(imgs_n_name[n_train: n_train + n_test], imgs_ab_name[an_train: an_train + an_test])
        _y_test = np.append(label_n[n_train: n_train + n_test], label_ab[an_train: an_train + an_test])
        num_train = len(_X_train)
        num_test = len(_X_test)
        train_idx = np.random.permutation(num_train)
        test_idx = np.random.permutation(num_test)
        _X_train = _X_train[train_idx]
        _y_train = _y_train[train_idx]
        _X_test =_X_test[test_idx]
        _y_test = _y_test[test_idx]

        if channel == 1:
            for a in _X_train:
                img = cv2.imread(a, flags = 0)
                img = cv2.resize(img, (enlr_size, enlr_size), interpolation=cv2.INTER_CUBIC)
                imgs_train.append(img)

            for b in _X_test:
                img = cv2.imread(b, flags=0)
                img = cv2.resize(img, (enlr_size, enlr_size), interpolation=cv2.INTER_CUBIC)
                imgs_test.append(img)

            train_n = np.shape(imgs_train)[0]
            test_n = np.shape(imgs_test)[0]
            imgs_train = np.reshape(imgs_train, [train_n, enlr_size, enlr_size, 1])
            imgs_test = np.reshape(imgs_test, [test_n, enlr_size, enlr_size, 1])

        else:
            for a in _X_train:
                    img = cv2.cvtColor(cv2.imread(a, flags=1), cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (enlr_size, enlr_size), interpolation=cv2.INTER_CUBIC)
                    imgs_train.append(img)
            for b in _X_test:
                    img = cv2.cvtColor(cv2.imread(b, flags=1), cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (enlr_size, enlr_size), interpolation=cv2.INTER_CUBIC)
                    imgs_test.append(img)

        imgs_train, imgs_test = self.preprocessing(imgs_train, imgs_test)
        print("mode: ", mode)
        print("Train images: ", np.shape(imgs_train))
        print("Test images: ", np.shape(imgs_test))

        return imgs_train, _y_train, imgs_test, _y_test

class batch_tower:

    def __init__(self, train, train_label, test, test_label, batch_size):
        self.self = self
        self.train = train
        self.train_label = train_label
        self.test = test
        self.test_label = test_label
        self.batch_size = batch_size

        if len(self.train) % self.batch_size == 0:
            self.n_train_batch = len(self.train)//self.batch_size
        else:
            self.n_train_batch = (len(self.train) // self.batch_size) + 1

        if len(self.test) % self.batch_size == 0:
            self.n_test_batch = len(self.test) // self.batch_size
        else:
            self.n_test_batch = (len(self.test) // self.batch_size) + 1

        self.iter_train = 0
        self.iter_test = 0

    def get_total_batch(self):
        return self.n_train_batch, self.n_test_batch

    def train_nb(self):
        if self.iter_train == (self.n_train_batch -1):
            train_xs = self.train[self.batch_size*-1: , :, :, :]
            train_ys = self.train_label[self.batch_size*-1:]
        else:
            train_xs = self.train[self.iter_train * self.batch_size: self.iter_train * self.batch_size + self.batch_size,:,:,:]
            train_ys = self.train_label[self.iter_train * self.batch_size: self.iter_train * self.batch_size + self.batch_size]

        self.iter_train += 1
        if self.iter_train == (self.n_train_batch):
            self.iter_train =0

        return train_xs, train_ys

    def test_nb(self):
        if self.iter_test == (self.n_test_batch -1):
            test_xs = self.test[self.batch_size*-1: , :, :, :]
            test_ys = self.test[self.batch_size*-1:]
        else:
            test_xs = self.test[self.iter_test * self.batch_size: self.iter_test * self.batch_size + self.batch_size,:,:,:]
            test_ys = self.test_label[self.iter_test * self.batch_size: self.iter_test * self.batch_size + self.batch_size]

        self.iter_test += 1
        if self.iter_test == (self.n_test_batch):
            self.iter_test =0

        return test_xs, test_ys