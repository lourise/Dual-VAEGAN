import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import util
import classifier as classifier
import classifier2 as classifier2
import sys
import model as model
import numpy as np
import semantic2lable as s2l
import itertools
from scipy import io as sio

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i

    return mapped_label

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)  # x_norm:64*1  x:64*2048
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)  # y_norm:1*450  y:450*2048
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    if y is None:  # dist reduce the diagonal matrix of dist
        dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)  # rescale the dist to [0, inf]

def loadPretrainedMain(netS, savePost):
    print('Loading pretrained Mainnet......')
    path = './checkpoint/'
    netS.load_state_dict(torch.load(path+savePost))
    return netS

def train(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = util.DATA_LOADER(opt)  # get train data
    opt['attSize'] = data.attribute.shape[1]
    opt['nz'] = opt['latent_feature_size']
    opt['device'] = device

    input_res = torch.FloatTensor(opt['batch_size'], opt['resSize']).to(device)  # batch_size*2048 pretrained image features?
    input_att = torch.FloatTensor(opt['batch_size'], opt['attSize']).to(device)  # batch_size*312
    noise = torch.FloatTensor(opt['batch_size'], opt['nz']).to(device)  # batch_size*312  generate the noise vectors
    unseen_noise = torch.FloatTensor(opt['batch_size'], opt['nz']).to(device)
    one = torch.tensor(1, dtype=torch.float).to(device)  # tensor number: 1
    mone = (one * -1).to(device)  # number: -1
    input_label = torch.LongTensor(opt['batch_size']).to(device)  # label
    input_label_ori = torch.LongTensor(opt['batch_size']).to(device)  # label

    unseen_res = torch.FloatTensor(opt['batch_size'], opt['resSize']).to(device)  # batch_size*2048 pretrained image features?
    unseen_att = torch.FloatTensor(opt['batch_size'], opt['attSize']).to(device)
    unseen_label = torch.LongTensor(opt['batch_size']).to(device)  # label
    unseen_label_ori = torch.LongTensor(opt['batch_size']).to(device)  # label
    def sample():  # get a batch of seen class data and attributes
        batch_feature, batch_label, batch_att = data.next_batch(opt['batch_size'])
        input_res.copy_(batch_feature)
        input_att.copy_(batch_att)
        input_label_ori.copy_(batch_label)
        input_label.copy_(util.map_label(batch_label, data.seenclasses))


    def sample_unseen():  # get a batch of unseen classes data and attributes
        batch_feature, batch_label, batch_att = data.next_batch_unseen(opt['batch_size'])
        unseen_res.copy_(batch_feature)
        unseen_att.copy_(batch_att)
        unseen_label_ori.copy_(batch_label)
        unseen_label.copy_(util.map_label(batch_label, data.unseenclasses))

    if opt['manualSeed'] is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt['manualSeed'])
    random.seed(opt['manualSeed'])
    torch.manual_seed(opt['manualSeed'])  # random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt['manualSeed'])
    cudnn.benchmark = True


    attE = model.Encoder_Att(opt).to(device)

    attD = model.Decoder_Att(opt).to(device)

    # initialize generator and discriminator
    netG = model.MLP_G(opt).to(device)  # initialize G and decoder
    print(netG)

    netD = model.MLP_D(opt).to(device)  # initialize D
    print(netD)

    netD2 = model.MLP_D2(opt).to(device)
    print(netD2)

    netE = model.Encoder(opt).to(device)  # initialize Encoder
    print(netE)

    netAlign = model.Discriminator_align(opt).to(device)


    logsoftmax = nn.LogSoftmax(dim=1)
    # classification loss, Equation (4) of the paper
    cls_criterion = nn.NLLLoss().to(device)


    netS = model.MLP_V2S(opt).to(device)
    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt['lr'], betas=(0.5, 0.999))
    optimizerD2 = optim.Adam(netD2.parameters(), lr=opt['lr'], betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt['lr'], betas=(0.5, 0.999))
    optimizerE = optim.Adam(netE.parameters(), lr=opt['lr'], betas=(0.5, 0.999))
    optimizerAE = optim.Adam(attE.parameters(), lr=opt['lr'], betas=(0.5, 0.999))
    optimizerAD = optim.Adam(attD.parameters(), lr=opt['lr'], betas=(0.5, 0.999))
    optimizerAlign = optim.Adam(netAlign.parameters(), lr=opt['lr'], betas=(0.5, 0.999))


    if opt['dataset']=='FLO':
        optimizerS = optim.Adam(netS.parameters(), lr=opt['lr']*1.5, betas=(0.5, 0.999))
    else:
        optimizerS = optim.Adam(netS.parameters(), lr=opt['lr'], betas=(0.5, 0.999))
    reg_criterion = nn.MSELoss().to(device)
    cro_criterion = nn.CrossEntropyLoss().to(device)
    binary_cross_entropy_crition = nn.BCELoss().to(device)
    nll_criterion = nn.NLLLoss().to(device)
    KL_criterion = nn.KLDivLoss().to(device)

    def getTestUnseenAcc():
        fake_unseen_attr = netS(Variable(data.test_unseen_feature.cuda(), volatile=True))
        dist = pairwise_distances(fake_unseen_attr.data, data.attribute[data.unseenclasses].cuda())  # range 50
        pred_idx = torch.min(dist, 1)[1]  # relative pred
        pred = data.unseenclasses[pred_idx.cpu()]  # map relative pred to absolute pred
        acc = sum(pred == data.test_unseen_label) / data.test_unseen_label.size()[0]
        print('Test Unseen Acc: {:.2f}%'.format(acc * 100))
        return logsoftmax(Variable(dist.cuda())).data

    def getTestAllAcc():
        fake_unseen_attr = netS(Variable(data.test_unseen_feature.cuda(), volatile=True))  #
        dist1 = pairwise_distances(fake_unseen_attr.data, data.attribute.cuda())  # 2967*200
        pred_idx = torch.min(dist1, 1)[1]  # absolute pred  2967
        acc_unseen = sum(pred_idx.cpu() == data.test_unseen_label).float() / data.test_unseen_label.size()[0]

        fake_seen_attr = netS(Variable(data.test_seen_feature.cuda(), volatile=True))
        dist2 = pairwise_distances(fake_seen_attr.data, data.attribute.cuda())  # range 200
        pred_idx = torch.min(dist2, 1)[1]  # absolute pred
        acc_seen = sum(pred_idx.cpu() == data.test_seen_label).float() / data.test_seen_label.size()[0]

        if (acc_seen == 0) or (acc_unseen == 0):
            H = 0
        else:
            H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
        print('Forward Seen:{:.2f}%, Unseen:{:.2f}%, H:{:.2f}%'.format(acc_seen * 100, acc_unseen * 100, H * 100))
        return logsoftmax(Variable(dist1.cuda())).data, logsoftmax(Variable(dist2.cuda())).data

    def caculateCosineSim(predAtt, allAtt):
        sims = torch.zeros((predAtt.shape[0], allAtt.shape[0])).to(device)
        for i, att in enumerate(predAtt):
            for j in range(allAtt.shape[0]):
                sims[i, j] = torch.nn.CosineSimilarity(dim=0)(att, allAtt[j])
        return sims

    modelStr = {'CUB': 'netS_CUB_Acc415900_03_15_11_23.pth', 'FLO': 'netS_FLO_Acc269300_03_15_10_26.pth',
                'SUN': 'netS_SUN_Acc457600_03_15_10_44.pth',
                'AWA1': 'netS_AWA1_Acc450100_03_15_10_58.pth', 'AWA2': 'netS_AWA2_Acc450100_03_15_10_58.pth',
                'APY': 'netS_APY_Acc203100_03_15_11_08.pth'}
    # if opt['use_pretrain_s'] == 1:
    #     netS = loadPretrainedMain(netS, modelStr[opt['dataset']])
    # else:
    #     netS.train()
    #     for epoch in range(opt['nets_epoch']):
    #         for i in range(0, data.ntrain, opt['batch_size']):
    #             optimizerS.zero_grad()
    #             sample()
    #             input_resv = Variable(input_res)
    #             input_attv = Variable(input_att)
    #             attv = Variable(data.most_sim_att[input_label_ori].to(device))
    #             pred = netS(input_resv)
    #             # sim_self = torch.nn.CosineSimilarity(dim=1)(pred, input_attv)
    #             # sim_most = torch.nn.CosineSimilarity(dim=1)(pred, attv)
    #             # ones = torch.ones_like(sim_most).to(device)
    #             # tmp = (sim_most > sim_self).sum()
    #             # loss_sim = (ones - sim_self).sum()
    #             mse_loss = reg_criterion(pred, input_attv)
    #             loss = mse_loss  # + loss_sim
    #             loss.backward()
    #             optimizerS.step()
    #         # print("epoch:%d, loss:%.4f" % (epoch, loss.item()))
    #         print(100 * '-')
    #         print("epoch:%d, mse_loss:%.4f" % (epoch, mse_loss))
    #         _ = getTestAllAcc()
    #     for p in netS.parameters():
    #         p.requires_grad = False
    #     netS.eval()
    #     os.makedirs('./checkpoint', exist_ok=True)
    #     torch.save(netS.state_dict(), './checkpoint/' + modelStr[opt['dataset']])
    # pretrain_cls = classifier.CLASSIFIER(data, opt, 0.001, 0.5, 100, 100)  # load pretrained model
    # for p in pretrain_cls.model.parameters():  # set requires_grad to False
    #     p.requires_grad = False
    # pretrain_cls.model.eval()
    # netS = loadPretrainedMain(netS, modelStr[opt['dataset']])
    if opt['generalized']:
        opt['gzsl_unseen_output'], opt['gzsl_seen_output'] = getTestAllAcc()
        with torch.no_grad():
            opt['fake_test_seen_attr'] = netS(
                data.test_seen_feature.cuda()).data  # generate the corresponding fake_attr
            opt['fake_test_unseen_attr'] = netS(data.test_unseen_feature.cuda()).data
    else:
        opt['gzsl_unseen_output'] = getTestUnseenAcc()
        with torch.no_grad():
            opt['fake_test_attr'] = netS(data.test_unseen_feature.cuda()).data

    def generate_syn_feature(netG, classes, attribute, num):  # only generate the unseen feature
        nclass = classes.size(0)
        syn_feature = torch.FloatTensor(nclass * num, opt['resSize'])
        syn_label = torch.LongTensor(nclass * num)
        syn_att = torch.FloatTensor(num, opt['attSize']).to(device)
        syn_noise = torch.FloatTensor(num, opt['nz']).to(device)
        with torch.no_grad():
            for i in range(nclass):
                iclass = classes[i]
                iclass_att = attribute[iclass]
                syn_att.copy_(iclass_att.repeat(num, 1))
                syn_noise.normal_(0, 1)  # directly sample noise from normal distribution and input the noise set Z into the G to get the newly generated features
                output = netG(syn_noise, syn_att)
                syn_feature.narrow(0, i * num, num).copy_(
                    output.data.cpu())  # narrow method is to get some dimension data
                syn_label.narrow(0, i * num, num).fill_(iclass)

        return syn_feature, syn_label

    def generate_seen_latent_feature(netG, netE, classes, attribute, num):  # only generate the unseen latent feature
        nclass = classes.size(0)
        syn_feature = torch.FloatTensor(nclass * num, opt['nz'])
        syn_label = torch.LongTensor(nclass * num)
        syn_att = torch.FloatTensor(num, opt['attSize']).to(device)
        syn_noise = torch.FloatTensor(num, opt['nz']).to(device)
        with torch.no_grad():
            for i in range(nclass):
                iclass = classes[i]
                iclass_att = attribute[iclass]
                syn_att.copy_(iclass_att.repeat(num, 1))
                syn_noise.normal_(0, 1)  # directly sample noise from normal distribution and input the noise set Z into the G to get the newly generated features
                output = netG(syn_noise, syn_att)
                output_mu, output_logvar = netE(output)
                output = reparameterize(output_mu, output_logvar)
                syn_feature.narrow(0, i * num, num).copy_(
                    output.data.cpu())  # narrow method is to get some dimension data
                syn_label.narrow(0, i * num, num).fill_(iclass)

        return syn_feature, syn_label

    def generate_unseen_latent_feature(netE, classes, attribute, num):  # only generate the unseen latent feature
        nclass = classes.size(0)
        syn_feature = torch.FloatTensor(nclass * num, opt['nz'])
        syn_label = torch.LongTensor(nclass * num)
        syn_att = torch.FloatTensor(num, opt['attSize']).to(device)
        syn_noise = torch.FloatTensor(num, opt['nz']).to(device)
        with torch.no_grad():
            for i in range(nclass):
                iclass = classes[i]
                iclass_att = attribute[iclass]
                syn_att.copy_(iclass_att.repeat(num, 1))
                # syn_noise.normal_(0, 1)  # directly sample noise from normal distribution and input the noise set Z into the G to get the newly generated features
                # output = netG(syn_noise, syn_att)
                output_mu, output_logvar = netE(syn_att)
                output = reparameterize(output_mu, output_logvar)
                syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())  # narrow method is to get some dimension data
                syn_label.narrow(0, i * num, num).fill_(iclass)

        return syn_feature, syn_label

    def generate_syn_feature_with_grad(netG, classes, attribute, num):
        nclass = classes.size(0)  # 150
        # syn_feature = torch.FloatTensor(nclass*num, opt['resSize'])
        syn_label = torch.LongTensor(nclass * num).to(device)
        syn_att = torch.FloatTensor(nclass * num, opt['attSize']).to(device)
        syn_noise = torch.FloatTensor(nclass * num, opt['nz']).to(device)

        syn_noise.normal_(0, 1)
        for i in range(nclass):
            iclass = classes[i]  # seen_classes
            iclass_att = attribute[iclass]
            syn_att.narrow(0, i * num, num).copy_(iclass_att.repeat(num, 1))  # 3000*312  0:row
            syn_label.narrow(0, i * num, num).fill_(iclass)
        syn_feature = netG(Variable(syn_noise), Variable(syn_att))
        return syn_feature, syn_label.cpu()

    def calc_gradient_penalty(netD, real_data, fake_data, input_att):
        # print real_data.size()
        alpha  = torch.rand(opt['batch_size'], 1)
        alpha = alpha.expand(real_data.size()).to(device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)  # get new input base on the real and fake features
        interpolates = interpolates.to(device)
        interpolates = Variable(interpolates, requires_grad=True)
        disc_interpolates = netD(interpolates, input_att)  #

        ones = torch.ones(disc_interpolates.size()).to(device)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,  # WGAN penalty
                                  grad_outputs=ones,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty


    def calc_gradient_penalty_without_att(netD2, real_data, fake_data, att):
        # print real_data.size()
        alpha  = torch.rand(opt['batch_size'], 1)
        alpha = alpha.expand(real_data.size()).to(device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)  # get new input base on the real and fake features
        interpolates = interpolates.to(device)
        interpolates = Variable(interpolates, requires_grad=True)
        disc_interpolates = netD2(interpolates, att)  #

        ones = torch.ones(disc_interpolates.size()).to(device)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,  # WGAN penalty
                                  grad_outputs=ones,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty

    def reparameterize(mu, logvar):
        sigma = torch.exp(logvar/2)
        eps = torch.cuda.FloatTensor(logvar.size()[0],1).normal_(0,1)
        eps  = eps.expand(sigma.size())
        return mu + sigma*eps

    def KL_distance(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def Align_distance(f_mu, f_logvar, att_mu, att_logvar):  # align the distribution in the third space
        s = torch.sqrt(torch.sum((f_mu - att_mu) ** 2, dim=1) + \
                   torch.sum((torch.sqrt(f_logvar.exp()) - torch.sqrt(att_logvar.exp())) ** 2, dim=1))
        return s.sum()

    # train a classifier on seen classes, obtain \theta of Equation (4)
    mse = nn.MSELoss().to(device)
    # freeze the classifier during the optimization
    best_H_cls = 0.0
    ones = torch.ones_like(input_label).type(torch.float).to(device)
    zeros = torch.zeros_like(input_label).type(torch.float).to(device)
    # pretrain_cls.model.eval()
    hyper_beta = {0.8, 1.2, 1.5, 2.0}
    hyper_distance = {5.13, 8.13, 12.13, 15.13}
    hyper_cycle = {1000, 3000, 5000, 7000}
    best_H = []
    tmp_H = []

# for (beta, distance, cycle) in itertools.product(hyper_beta,hyper_distance,hyper_cycle):
    if not os.path.exists('./checkpoint/' + 'pretrained_cadavae_' + opt['dataset'] + '-' + str(opt['pretrained_num']) + '.pth'):

        for ae_epoch in range(opt['pretrained_num']):
                # warm up parameter()
            f1 = 1.0 * (ae_epoch- opt['warmup']['cross_reconstruction']['start_epoch']) / \
                 (1.0 * (opt['warmup']['cross_reconstruction']['end_epoch'] - opt['warmup']['cross_reconstruction']['start_epoch']))
            f1 = f1 * (1.0 * opt['warmup']['cross_reconstruction']['factor'])
            cross_reconstruction_factor = torch.cuda.FloatTensor([min(max(f1, 0), opt['warmup']['cross_reconstruction']['factor'])])
            # (epoch-0)/(93-0)*0.25
            f2 = 1.0 * (ae_epoch - opt['warmup']['beta']['start_epoch']) /\
                 (1.0 * (opt['warmup']['beta']['end_epoch'] - opt['warmup']['beta']['start_epoch']))
            f2 = f2 * (1.0 * opt['warmup']['beta']['factor'])
            beta = torch.cuda.FloatTensor([min(max(f2, 0), opt['warmup']['beta']['factor'])])
            # (epoch-6)/(22-6)*8.13
            f3 = 1.0 * (ae_epoch - opt['warmup']['distance']['start_epoch']) /\
                 (1.0 * (opt['warmup']['distance']['end_epoch'] - opt['warmup']['distance']['start_epoch']))
            f3 = f3 * (1.0 * opt['warmup']['distance']['factor'])
            distance_factor = torch.cuda.FloatTensor([min(max(f3, 0), opt['warmup']['distance']['factor'])])

            f4 = 1.0 * (ae_epoch - opt['warmup']['cycle']['start_epoch']) / \
                 (1.0 * (opt['warmup']['cycle']['end_epoch'] - opt['warmup']['cycle']['start_epoch']))
            f4 = f4 * (1.0 * opt['warmup']['cycle']['factor'])
            cycle_factor = torch.cuda.FloatTensor([min(max(f4, 0), opt['warmup']['cycle']['factor'])])

            for i in range(0, data.ntrain, opt['batch_size']):

                optimizerAD.zero_grad()
                optimizerAE.zero_grad()
                optimizerE.zero_grad()
                optimizerG.zero_grad()

                sample()
                att_mu, att_logvar = attE(input_att)
                latent_att = reparameterize(att_mu, att_logvar)
                recon_att = attD(latent_att)
                att_reconstruct_to_feature = netG(latent_att, input_att)
                att_self_recon_loss = binary_cross_entropy_crition(recon_att, input_att)
                att_cross_recon_loss = binary_cross_entropy_crition(att_reconstruct_to_feature, input_res)
                att_KL_loss = KL_distance(att_mu, att_logvar)
                loss_att = att_self_recon_loss + beta * att_KL_loss + cross_reconstruction_factor * att_cross_recon_loss

                real_mu, real_logvar = netE(input_res)
                latent_feature = reparameterize(real_mu, real_logvar)
                reconstruct_feature = netG(latent_feature, input_att)
                # KLD = KL_distance(real_mu, real_logvar)
                # distribution = torch.FloatTensor(opt['batch_size'], opt['nz']).requires_grad(False)
                # distribution = distribution.copy_(noise).requires_grad(False)
                feature_KL_dis = KL_distance(real_mu, real_logvar)
                reconstruct_loss = binary_cross_entropy_crition(reconstruct_feature, input_res)
                feature_reconstruct_to_att = attD(latent_feature)
                feature_cross_recon_loss = binary_cross_entropy_crition(feature_reconstruct_to_att, input_att)
                loss_feature = reconstruct_loss + beta * feature_KL_dis + cross_reconstruction_factor * feature_cross_recon_loss

                align_loss = Align_distance(real_mu, real_logvar, att_mu, att_logvar)

                re_reconstruct_att_mu, re_reconstruct_att_logvar = attE(recon_att)
                re_reconstruct_latent_att = reparameterize(re_reconstruct_att_mu, re_reconstruct_att_logvar)
                re_reconstruct_att = attD(re_reconstruct_latent_att)
                cycle_att_loss = binary_cross_entropy_crition(re_reconstruct_att, input_att)  # + reg_criterion(re_reconstruct_att, recon_att.data)

                re_reconstruct_latent_feature_mu, re_reconstruct_latent_feature_logvar = netE(reconstruct_feature)
                re_reconstruct_latent_feature = reparameterize(re_reconstruct_latent_feature_mu,
                                                               re_reconstruct_latent_feature_logvar)
                re_reconstruct_feature = netG(re_reconstruct_latent_feature, input_att)
                cycle_loss = binary_cross_entropy_crition(re_reconstruct_feature, input_res)  # + reg_criterion(re_reconstruct_feature, reconstruct_feature.data)

                VAE_loss = loss_feature + loss_att + distance_factor * align_loss + cycle_factor * (cycle_att_loss + cycle_loss)

                VAE_loss.backward()

                optimizerG.step()
                optimizerE.step()
                optimizerAE.step()
                optimizerAD.step()

        torch.save({
            'G_state_dict': netG.state_dict(),
            'E_state_dict': netE.state_dict(),
            'AE_state_dict': attE.state_dict(),
            'AD_state_dict': attD.state_dict(),
            # 'netS_state_dict': netS.state_dict(),
        }, './checkpoint/' + 'pretrained_cadavae_' + opt['dataset'] + '-' + str(opt['pretrained_num']) + '.pth')

    else:
        pretrained_path = torch.load('./checkpoint/' + 'pretrained_cadavae_' + opt['dataset'] + '-' + str(opt['pretrained_num']) + '.pth')
        netG.load_state_dict(pretrained_path['G_state_dict'])
        netE.load_state_dict(pretrained_path['E_state_dict'])
        attD.load_state_dict(pretrained_path['AD_state_dict'])
        attE.load_state_dict(pretrained_path['AE_state_dict'])
    # for epoch in range(80):  # pretrain VAE
    #
    #     for i in range(0, data.ntrain, opt['batch_size']):
    #         sample()
    #         optimizerE.zero_grad()
    #         optimizerG.zero_grad()
    #         latent_mu, latent_logvar = netE(input_res)
    #         latent = reparameterize(latent_mu, latent_logvar)
    #         re = netG(latent, input_att)
    #         kl_loss = KL_distance(latent_mu, latent_logvar)
    #         re_loss = reg_criterion(input_res, re)
    #         loss = kl_loss+ re_loss
    #         loss.backward()
    #         optimizerG.step()
    #         optimizerE.step()



    for epoch in range(opt['nepoch']):  # after 5 times update of discriminator will update generator once

        for i in range(0, data.ntrain, opt['batch_size']):
            # update the parameters of discriminator.
            for p in netD.parameters():
                p.requires_grad = True
            for p in netD2.parameters():
                p.requires_grad = True

            for iter_d in range(5):  # 5
                sample()  # get samples
                optimizerD.zero_grad()
                optimizerD2.zero_grad()

                criticD_real = netD(input_res, input_att)  # real samples for D
                # criticD_real = binary_cross_entropy_crition(criticD_real, ones)
                criticD_real = criticD_real.mean()

                noise.normal_(0, 1)  # get seen class noise
                fake = netG(noise, input_att)  # generate seen classes fake features for D
                criticD_fake = netD(fake.detach(), input_att)  # discriminate the fake feature and get loss
                # criticD_fake = binary_cross_entropy_crition(criticD_fake, zeros)
                criticD_fake = criticD_fake.mean()
                # feature_reconstruct_mu, feature_reconstruct_logvar = netE(input_res)
                # reconstruct_latent_feature = reparameterize(feature_reconstruct_mu, feature_reconstruct_logvar)
                # criticD_real_reconstruct = netG(reconstruct_latent_feature, input_att)
                # criticD_real_reconstruct  = netD(criticD_real_reconstruct, input_att)
                # criticD_real_reconstruct = criticD_real_reconstruct.mean()



                criticD2_real = netD2(input_att, input_att)
                criticD2_real = criticD2_real.mean()
                criticD2_real_mu, critiD2_real_logvar = attE(input_att)
                criticD2_real_latent_feature = reparameterize(criticD2_real_mu, critiD2_real_logvar)
                real_construct = attD(criticD2_real_latent_feature)
                criticD2_real_reconstruct = netD2(real_construct, input_att)
                criticD2_real_reconstruct = criticD2_real_reconstruct.mean()
                fake_att = attD(noise)
                criticD2_fake = netD2(fake_att.detach(), input_att)
                criticD2_fake = criticD2_fake.mean()

                gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)  # weight penalty?

                gradient_penalty_D2 = calc_gradient_penalty_without_att(netD2, input_att, fake_att.data, input_att)

                D_cost = criticD_fake - criticD_real + gradient_penalty  # total loss

                D2_cost = criticD2_fake + gradient_penalty_D2 - criticD2_real  # + criticD2_real_reconstruct

                D_loss = D_cost + D2_cost

                D_loss.backward()
                optimizerD.step()
                optimizerD2.step()

            # update the parameters of generator.
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = False  # avoid computation

            for p in netD2.parameters():  # reset requires_grad
                p.requires_grad = False  # avoid computation

            optimizerG.zero_grad()
            optimizerE.zero_grad()
            optimizerAE.zero_grad()
            optimizerAD.zero_grad()
            optimizerAlign.zero_grad()


            sample()
            noise.normal_(0, 1)

            att_mu, att_logvar = attE(input_att)
            latent_att = reparameterize(att_mu, att_logvar)
            recon_att = attD(latent_att)
            self_recon_loss = binary_cross_entropy_crition(recon_att, input_att)
            att_reconstruct_to_feature = netG(latent_att, input_att)
            att_cross_recon_loss = binary_cross_entropy_crition(att_reconstruct_to_feature, input_res)
            att_KL_loss = - KL_distance(att_mu, att_logvar)
            # loss_att = self_recon_loss + beta * att_KL_loss  + att_cross_recon_loss * opt['warmup']['cross_reconstruction']['factor']
            loss_att = self_recon_loss + opt['warmup']['beta']['factor'] * att_KL_loss  + att_cross_recon_loss * opt['warmup']['cross_reconstruction']['factor']

            re_reconstruct_att_mu, re_reconstruct_att_logvar = attE(recon_att)
            re_reconstruct_latent_att = reparameterize(re_reconstruct_att_mu, re_reconstruct_att_logvar)
            re_reconstruct_att = attD(re_reconstruct_latent_att)
            cycle_att_loss = binary_cross_entropy_crition(re_reconstruct_att, input_att)  # + binary_cross_entropy_crition(re_reconstruct_att, recon_att.data)
            # cycle_latent_att_loss = reg_criterion(latent_att, re_reconstruct_latent_att)


            real_mu, real_logvar = netE(input_res)
            latent_feature = reparameterize(real_mu, real_logvar)
            reconstruct_feature = netG(latent_feature, input_att)
            feature_KL_dis = KL_distance(real_mu, real_logvar)
            reconstruct_loss = binary_cross_entropy_crition(reconstruct_feature, input_res)
            feature_reconstruct_to_att = attD(latent_feature)
            feature_cross_recon_loss = binary_cross_entropy_crition(feature_reconstruct_to_att, input_att)
            # loss_feature = reconstruct_loss + beta * feature_KL_dis + feature_cross_recon_loss * opt['warmup']['cross_reconstruction']['factor']
            loss_feature = reconstruct_loss + opt['warmup']['beta']['factor'] * feature_KL_dis + feature_cross_recon_loss * opt['warmup']['cross_reconstruction']['factor']

            re_reconstruct_latent_feature_mu, re_reconstruct_latent_feature_logvar = netE(reconstruct_feature)
            re_reconstruct_latent_feature = reparameterize(re_reconstruct_latent_feature_mu, re_reconstruct_latent_feature_logvar)
            re_reconstruct_feature = netG(re_reconstruct_latent_feature, input_att)
            cycle_loss = binary_cross_entropy_crition(re_reconstruct_feature, input_res)  # + binary_cross_entropy_crition(re_reconstruct_feature, reconstruct_feature.data)
            # cycle_latent_feature_loss = reg_criterion(re_reconstruct_latent_feature, latent_feature)

            align_loss = Align_distance(real_mu, real_logvar, att_mu, att_logvar)
            # vae_loss = loss_feature + loss_att + align_loss * distance + cycle_loss * cycle + cycle_att_loss * cycle  # + cycle_latent_att_loss * 2 + cycle_latent_feature_loss * 2
            vae_loss = 2*(loss_feature + loss_att + align_loss * opt['warmup']['distance']['factor']) + cycle_loss * 10000 + cycle_att_loss * 10000  # + cycle_latent_att_loss * 2 + cycle_latent_feature_loss * 2

            # vae_loss.backward(retain_graph=True)
            # noise.normal_(0, 1)
            fake = netG(noise, input_att)
            # criticD_real = netD(input_res, input_att)  # of no use
            criticG_fake = netD(fake, input_att)
            criticG_fake = criticG_fake.mean()
            # criticG_fake2 = netD(reconstruct_feature, input_att)
            # criticG_fake2 = criticG_fake2.mean()
            G_cost = - criticG_fake * opt['wgan_weight']  # - change minimize target to maximize the target- criticG_fake2

            fake_att = attD(noise)
            criticDe_fake = netD2(fake_att, input_att)
            criticDe_fake = criticDe_fake.mean()
            # criticDe_fake2 = netD2(recon_att, input_att)
            # criticDe_fake2 = criticDe_fake2.mean()
            D2_cost = - criticDe_fake * opt['wgan_weight']  # - criticDe_fake2 * 1

            #
            # origin_feature_latent_feature = netAlign(input_res, latent_feature)
            # fake_latent_feature_rebuild_feature = netAlign(fake, noise)
            # BiGAN_loss = binary_cross_entropy_crition(origin_feature_latent_feature, ones) - binary_cross_entropy_crition(fake_latent_feature_rebuild_feature, zeros)
            # classification loss
            # c_errG = cls_criterion(pretrain_cls.model(fake), input_label)

            loss = G_cost + vae_loss + D2_cost  # + 0.7 * BiGAN_loss  # + loss_feature + loss_att + 8.13 * align_loss

            loss.backward()
            optimizerAE.step()
            optimizerAD.step()
            optimizerG.step()
            optimizerE.step()
            optimizerAlign.step()

        print('EP[%d/%d]*******************************************************************' % (epoch, opt['nepoch']))
        print('G_cost: %.4f, D2_loss: %.4f, vae_loss: %.4f, loss: %.4f' % (G_cost.item(), D2_cost, vae_loss.item(), loss.item()))
        # print('tmp_seen:%d, tmp_unseen:%d' % (tmp_seen, tmp_unseen))5867

        # evaluate the model, set G to evaluation mode
        if epoch >= 0:
            netG.eval()
            netE.eval()
            attE.eval()
            attD.eval()
            # Generalized zero-shot learning
            syn_unseen_feature, syn_unseen_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt['cls_syn_num'])  # 1500x2048 generate unseen classed feature
            syn_seen_feature, syn_seen_label = generate_syn_feature(netG, data.seenclasses, data.attribute, int(opt['cls_syn_num'] / 30))
            if opt['generalized']:
                # train_latent_feature_mu, train_latent_feature_logvar = netE(data.train_feature.cuda(device))
                # train_latent_feature = reparameterize(train_latent_feature_mu, train_latent_feature_logvar).data.cpu()
                train_X = torch.cat((data.train_feature, syn_unseen_feature, syn_seen_feature), 0)  # combine seen and unseen classes features
                train_Y = torch.cat((data.train_label, syn_unseen_label, syn_seen_label), 0)
                # train_X = torch.cat((data.train_feature, syn_unseen_feature),0)
                # train_Y = torch.cat((data.train_label, syn_unseen_label), 0)
                # if data.test_seen_feature.size()[-1] != opt['nz']:
                #     test_unseen_feature, _ = netE(data.test_unseen_feature.cuda(device))
                #     test_seen_feature, _ = netE(data.test_seen_feature.cuda(device))
                #     data.test_unseen_feature = test_unseen_feature
                #     data.test_seen_feature = test_seen_feature
                nclass = data.ntrain_class + data.ntest_class  # classes numbers
                v2s = s2l.Visual_to_semantic(opt, netS(train_X.cuda()).data.cpu(), train_Y, data, nclass, generalized=True)
                opt['gzsl_unseen_output'] = v2s.unseen_out
                opt['gzsl_seen_output'] = v2s.seen_out
                cls = classifier2.CLASSIFIER(opt, train_X, train_Y, data, nclass, _beta1=0.5, _nepoch=30, generalized=True)
                print(
                    'GZSL Classifier Seen Acc: {:.2f}%, Unseen Acc: {:.2f}%, H Acc: {:.2f}%'.format(cls.seen_cls * 100,
                                                                                                    cls.unseen_cls * 100,
                                                                                                    cls.H_cls * 100))

                print('GZSL Ensemble Seen Acc: {:.2f}%, Unseen Acc: {:.2f}%, H Acc: {:.2f}%'.format(
                    cls.seen_ensemble * 100, cls.unseen_ensemble * 100, cls.H_ensemble * 100))
                if cls.H_cls > best_H_cls:
                    best_H_cls = cls.H_cls
                    torch.save({'G_state_dict': netG.state_dict(),
                                'D_state_dict': netD.state_dict(),
                                'netS_state_dict': netS.state_dict(),
                                'H': cls.H_cls,
                                'gzsl_seen_accuracy': cls.seen_cls,
                                'gzsl_unseen_accuracy': cls.unseen_cls,
                                'cls': cls,
                                },
                               './checkpoint/' + 'gzsl_' + opt['dataset'] + '-' + str(epoch) + '.pth')
                    tmp_H.append([cls.seen_cls, cls.unseen_cls, cls.H_cls])
                    sio.savemat('./data/' + opt['dataset'] + '/fakeTestFeat.mat',
                                {'train_X': train_X.numpy(), 'train_Y': train_Y.numpy(),
                                 'test_seen_X': data.test_seen_feature.numpy(),
                                 'test_seen_Y': data.test_seen_label.numpy(),
                                 'test_unseen_X': data.test_unseen_feature.numpy(),
                                 'test_unseen_Y': data.test_unseen_label.numpy()})
            else:
                fake_syn_unseen_attr = netS(Variable(syn_unseen_feature.cuda(), volatile=True))[0]
                v2s = s2l.Visual_to_semantic(opt, fake_syn_unseen_attr.data.cpu(), syn_unseen_label, data,
                                             data.unseenclasses.size(0), generalized=False)
                opt.zsl_unseen_output = v2s.output
                cls = classifier2.CLASSIFIER(opt, syn_unseen_feature,
                                             util.map_label(syn_unseen_label, data.unseenclasses),
                                             data, data.unseenclasses.size(0), _beta1=0.5, _nepoch=25,
                                             generalized=False)
                print('ZSL Classifier: {:.2f}%'.format(cls.cls_acc * 100))
                print('ZSL Ensemble: {:.2f}%'.format(cls.ensemble_acc * 100))
            sys.stdout.flush()
            netG.train()
            netE.train()
            attE.train()
            attD.train()
    # best_H.append(tmp_H[-1])
    # print('Parameter setting: beta: {:}%, distance: {:}%, cycle: {:}%, best_acc: seen:{:.2f}%, unseen:{:.2f}%,'
    #       ' H:{:.2f}%'.format(beta, distance, cycle, tmp_H[-1][0]*100, tmp_H[-1][1]*100, tmp_H[-1][-1]*100))
    # tmp_H = []
    # best_H_cls = 0.0
    # sio.savemat('./best_acc.mat',
    #         {   'para':itertools.product(beta, distance, cycle),
    #             'acc':best_H,
    #         })