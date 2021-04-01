import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def map_label(label, classes):  # get the relative seen label
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label

class Logger(object):
    def __init__(self, fileN='Default.log'):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class DATA_LOADER(object):
    def __init__(self, opt):
        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

        self.feature_dim = self.train_feature.shape[1]
        self.att_dim = self.attribute.shape[1]
        self.text_dim = self.att_dim
        self.train_cls_num = self.seenclasses.shape[0]
        self.test_cls_num = self.unseenclasses.shape[0]
        self.tr_cls_centroid = np.zeros([self.seenclasses.shape[0], self.feature_dim], np.float32)  # centroid?
        for i in range(self.seenclasses.shape[0]):
            self.tr_cls_centroid[i] = np.mean(self.train_feature[torch.nonzero(self.train_mapped_label == i),:].numpy(), axis=0)  # the mean of each class

        real_proto = torch.zeros(self.train_cls_num, self.feature_dim)  # 450*2048
        for i in range(self.train_cls_num):  # 150
            sample_idx = (self.train_mapped_label == i).nonzero().squeeze()
            real_proto[i, :] = self.train_feature[sample_idx, :].mean(dim=0).reshape((1, -1))
        self.real_proto = real_proto

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt['dataroot'] + "/" + opt['dataset'] + "/" + opt['image_embedding'] + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt['dataroot'] + "/" + opt['dataset'] + "/" + opt['class_embedding'] + ".mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        if opt['preprocessing']:
            scaler = preprocessing.MinMaxScaler()

            _train_feature = scaler.fit_transform(feature[trainval_loc])
            _test_seen_feature = scaler.transform(feature[test_seen_loc])
            _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
            self.train_feature = torch.from_numpy(_train_feature).float()
            mx = self.train_feature.max()
            self.train_feature.mul_(1/mx)
            self.train_label = torch.from_numpy(label[trainval_loc]).long()
            self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
            self.test_unseen_feature.mul_(1/mx)
            self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
            self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
            self.test_seen_feature.mul_(1/mx)
            self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
            self.train_label = torch.from_numpy(label[trainval_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
            self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
            self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntest = self.test_unseen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)
        self.train_att = self.attribute[self.seenclasses].numpy()
        self.test_att  = self.attribute[self.unseenclasses].numpy()
        self.most_sim_att = torch.zeros_like(self.attribute)  # calculate the cosine distance of attributes? without using
        for i, att_a in enumerate(self.attribute):
            num1_sim = 0
            for j, att_b in enumerate(self.attribute):
                if i != j:
                    sim = torch.nn.CosineSimilarity(dim=0)(att_a, att_b)
                    if num1_sim < sim:
                        num1_sim = sim
                        num1_att = att_b
                else:
                    continue
            self.most_sim_att[i, :] = num1_att

        self.train_cls_num = self.ntrain_class
        self.test_cls_num  = self.ntest_class

    def next_batch(self, batch_size):  # get a batch of seen class sample, labels and attributes
        idx = torch.randperm(self.ntrain)[0:batch_size]  # randperm return a array from 0 to giving number
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    def next_batch_unseen(self, batch_size):  # get a batch of unseen class sample, labels and attributes
        idx = torch.randperm(self.ntest)[0:batch_size]  # randperm return a array from 0 to giving number
        batch_feature = self.test_unseen_feature[idx]
        batch_label = self.test_unseen_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att