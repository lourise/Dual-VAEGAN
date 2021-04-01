import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import util


class CLASSIFIER:
    # train_Y is interger   (data, opt, 0.001, 0.5, 50, 100, opt.pretrain_classifier)
    def __init__(self, data, opt, _lr=0.001, _beta1=0.5, _nepoch=100, _batch_size=100):
        self.data = data
        self.dataset = opt['dataset']
        self.train_X = data.train_feature
        self.train_Y = util.map_label(data.train_label, data.seenclasses)
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = data.seenclasses.size(0)
        self.input_dim = data.train_feature.shape[1]
        self.device = opt['device']
        self.model = LINEAR_LOGSOFTMAX(self.input_dim, self.nclass).to(self.device)
        self.model.apply(util.weights_init)
        self.criterion = nn.NLLLoss().to(self.device)

        self.input = torch.FloatTensor(_batch_size, self.input_dim).to(self.device)
        self.label = torch.LongTensor(_batch_size).to(self.device)

        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]
        self.pretrain_classifier = opt['pretrain_classifier']

        if self.pretrain_classifier == '':
            self.fit()
        else:
            print("loading the pretrained classifer...")
            self.model.load_state_dict(torch.load(self.pretrain_classifier))

    def fit(self):
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()
            acc = self.val(self.train_X, self.train_Y, self.data.seenclasses)
            print('epoch:%d, acc %.4f' % (epoch, acc))
            val_acc = self.val(self.data.test_seen_feature,
                               util.map_label(self.data.test_seen_label, self.data.seenclasses), self.data.seenclasses)
            print('val_acc:%.4f' % val_acc)

        if self.pretrain_classifier == '':
            import os
            os.makedirs('./checkpoint', exist_ok=True)
            torch.save(self.model.state_dict(), './checkpoint/cl_' + self.dataset + '.pth')

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0), torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]

    # test_label is integer
    def val(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            output = self.model(Variable(test_X[start:end].to(self.device), volatile=True))
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc = self.compute_per_class_acc(test_label, predicted_label, target_classes.size(0))
        return acc

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx)
        return acc_per_class.mean()


class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o  
