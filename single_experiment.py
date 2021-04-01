import os
import sys
import time
import argparse
import torch
from util import Logger
import CLSWGAN, cycle_vaegan, cycle_vaegan2
import VAEGAN
torch.cuda.set_device(3)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='APY')
parser.add_argument('--generalized', type=str2bool, default=True)
parser.add_argument('--pretrainedSC', type=str2bool, default=False)  # use the pretrained S,C or not
args = parser.parse_args()


os.makedirs('./log', exist_ok=True)
sys.stdout = Logger('log/('+args.dataset+')'+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'(wt-nets.log')


with open('single_experiment.py', encoding='UTF-8') as f:
    contents = f.read()
    print(contents)
f.close()
with open('cycle_vaegan2.py', encoding='UTF-8') as f:
    contents = f.read()
    print(contents)
f.close()


########################################
# the basic hyperparameters
########################################
hyperparameters = {
    'manualSeed': 9182,  # 9182  8894
    'cls_weight': 0.01,
    'wgan_weight': 10,
    # 'perceptual_weight': 0.0,
    'preprocessing': True,
    'generalized': True,
    'lr': 1e-4,
    'image_embedding': 'res101',
    'class_embedding': 'att_splits',
    'lambda': 10,
    'batch_size': 64,
    'resSize': 2048,
    'dataroot': './data',
    'classifier_lr': 0.0005,
    'latent_feature_size': 64,
    'activate_index': 0.2,
    'pretrained_num': 20,
    'warmup': {'beta': {'factor': 2.5, 'end_epoch': 93, 'start_epoch': 0},
               'cross_reconstruction': {'factor': 2.37, 'end_epoch': 75, 'start_epoch': 15},
               'distance': {'factor': 15.13, 'end_epoch': 22, 'start_epoch': 6},
                'cycle':{'factor': 1, 'end_epoch': 80, 'start_epoch': 50},
               },
}
if args.pretrainedSC:
    hyperparameters['use_pretrain_s'] = True
    hyperparameters['pretrain_classifier'] = './checkpoint/cl_' + args.dataset + '.pth'
else:
    hyperparameters['use_pretrain_s'] = False
    hyperparameters['pretrain_classifier'] = ''

# The training epochs for the final classifier, for early stopping, as determined on the validation spit
hyps = [
      {'dataset': 'AWA1', 'nets_epoch': 100, 'loss_syn_num': 30,  'netS_hid': 4096,  'ensemble_ratio': 1.5,  'cls_syn_num': 5000, 'cls_batch_size': 2000, 'nepoch': 100, 'netE_hid': 4096, 'netD_hid':1024,},
      {'dataset': 'AWA2', 'nets_epoch': 100, 'loss_syn_num': 30,  'netS_hid': 4096,  'ensemble_ratio': 1.5,  'cls_syn_num': 2400, 'cls_batch_size': 1650, 'nepoch': 100, 'netE_hid': 4096, 'netD_hid':1024,},
      {'dataset': 'CUB',  'nets_epoch': 100, 'loss_syn_num': 50,  'netS_hid': 8192,  'ensemble_ratio': 1.5,  'cls_syn_num': 450, 'cls_batch_size': 150, 'nepoch': 150, 'netE_hid': 4096, 'netD_hid':1024,},
      {'dataset': 'SUN', 'nets_epoch': 100, 'loss_syn_num': 20, 'netS_hid': 8192, 'ensemble_ratio': 1.6,  'cls_syn_num': 600, 'cls_batch_size': 300, 'nepoch':150, 'netE_hid': 4096, 'netD_hid':1024,},
      {'dataset': 'APY', 'nets_epoch': 100, 'loss_syn_num': 10,  'netS_hid': 8192, 'ensemble_ratio': 1.5, 'cls_syn_num': 3000, 'cls_batch_size': 600, 'nepoch': 150, 'netE_hid': 4096, 'netD_hid':1024,},
      {'dataset': 'FLO', 'nets_epoch': 100, 'loss_syn_num': 10, 'netS_hid': 4096, 'ensemble_ratio': 1.5, 'cls_syn_num': 450, 'cls_batch_size': 300, 'nepoch': 150, 'netE_hid': 4096, 'netD_hid':1024,},
      ]

##################################
# change some hyperparameters here
##################################
hyperparameters['dataset'] = args.dataset
hyperparameters['generalized'] = args.generalized
# train_steps
for hyp in hyps:
    if hyp['dataset'] == hyperparameters['dataset']:
        for k, v in hyp.items():
            hyperparameters[k] = v
        break
CLSWGAN.train(hyperparameters)
print()

