import torch.nn as nn
import torch
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



# discriminator
class MLP_D(nn.Module):
    def __init__(self, opt):
        super(MLP_D, self).__init__()
        # self.fc1 = nn.Linear(opt['resSize'] + opt['attSize'], 4096)  # origin
        self.fc1 = nn.Linear(opt['resSize'] + opt['attSize'], opt['netD_hid'])
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt['netD_hid'], 1)
        self.lrelu = nn.LeakyReLU(opt['activate_index'], True)
        self.sigoid = nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.fc1(h)
        h = self.lrelu(h)
        h = self.fc2(h)
        # h = self.sigoid(h)
        return h

class MLP_D2(nn.Module):
    def __init__(self, opt):
        super(MLP_D2, self).__init__()
        self.fc1 = nn.Linear(opt['attSize'], opt['netD_hid'])
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt['netD_hid'], 1)
        self.lrelu = nn.LeakyReLU(opt['activate_index'], True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)
        self.to(opt['device'])

    def forward(self, x, att):
        # h = torch.cat((x, att), 1)
        h = self.fc1(x)
        h = self.lrelu(h)
        h = self.fc2(h)
        # h = self.sigmoid(h)
        return h


class MLP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(opt['attSize'] + opt['latent_feature_size'], 4096)
        self.fc2 = nn.Linear(4096, opt['resSize'])
        self.lrelu = nn.LeakyReLU(opt['activate_index'], True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.sigmoid(self.fc2(h))
        return h

class MLP_V2S(nn.Module):
    def __init__(self, opt):
        super(MLP_V2S, self).__init__()
        self.fc1 = nn.Linear(opt['resSize'], opt['netS_hid'])
        self.fc2 = nn.Linear(opt['netS_hid'], opt['attSize'])
        self.fc3 = nn.Linear(opt['attSize'], opt['attSize'])
        self.fc4 = nn.Linear(opt['attSize'], opt['attSize'])
        self.lrelu = nn.LeakyReLU(opt['activate_index'], True)
        # self.lrelu = nn.ReLU(True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, res):
        h = self.lrelu(self.fc1(res))
        att_anchor = self.relu(self.fc2(h))  # anchor?
        return att_anchor


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(opt['resSize'], opt['netE_hid'])
        self.fc2 = nn.Linear(opt['netE_hid'], opt['latent_feature_size'])
        self.lrelu = nn.LeakyReLU(opt['activate_index'], True)
        self._mu = nn.Linear(opt['netE_hid'], opt['latent_feature_size'])
        self._logvar = nn.Linear(opt['netE_hid'], opt['latent_feature_size'])
        self.apply(weights_init)
        self.to(opt['device'])

    def forward(self, res):
        h = self.lrelu(self.fc1(res))
        # latent_feature = self.lrelu(self.fc2(h))

        mu = self._mu(h)
        logvar = self._logvar(h)

        return mu, logvar

class Encoder_Att(nn.Module):
    def __init__(self, opt):
        super(Encoder_Att, self).__init__()
        self.fc1 = nn.Linear(opt['attSize'], opt['netE_hid'])
        self.fc2 = nn.Linear(opt['netE_hid'], opt['latent_feature_size'])
        self.lrelu = nn.LeakyReLU(opt['activate_index'], True)

        self._mu = nn.Linear(opt['netE_hid'], opt['latent_feature_size'])
        self._logvar = nn.Linear(opt['netE_hid'], opt['latent_feature_size'])
        self.apply(weights_init)
        self.to(opt['device'])

    def forward(self, res):
        h = self.lrelu(self.fc1(res))
        # latent_feature = self.lrelu(self.fc2(h))
        mu = self._mu(h)
        logvar = self._logvar(h)

        return mu, logvar

class Decoder_Att(nn.Module):
    def __init__(self, opt):
        super(Decoder_Att, self).__init__()
        self.fc1 = nn.Linear(opt['latent_feature_size'], opt['netE_hid'])
        self.fc2 = nn.Linear(opt['netE_hid'], opt['attSize'])
        self.lrelu = nn.LeakyReLU(opt['activate_index'], True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

        self.to(opt['device'])

    def forward(self, res):
        h = self.lrelu(self.fc1(res))
        recon_feature = self.sigmoid(self.fc2(h))

        return recon_feature

class Discriminator_align(nn.Module):
    def __init__(self, opt):
        super(Discriminator_align, self).__init__()
        self.fc1 = nn.Linear(opt['latent_feature_size']+opt['resSize'], opt['netE_hid'])
        self.fc2 = nn.Linear(opt['netE_hid'], 1)
        self.lrelu = nn.LeakyReLU(opt['activate_index'], True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)
        self.to(opt['device'])

    def forward(self, res, latent_feature):
        h = torch.cat((res, latent_feature), dim=1)
        h = self.lrelu(self.fc1(h))
        h = self.sigmoid(self.fc2(h))
        return h

# class MLP_V2S_base(nn.Module):
#     def __init__(self, opt):
#         super(MLP_V2S_base, self).__init__()
#         self.fc1 = nn.Linear(opt['resSize'], opt['netS_hid'])
#         self.fc2 = nn.Linear(opt['netS_hid'], opt['attSize'])
#         self.lrelu = nn.LeakyReLU(0.2, True)
#         # self.lrelu = nn.ReLU(True)
#         self.relu = nn.ReLU(True)
#         self.apply(weights_init)
#
#     def forward(self, res):
#         h = self.lrelu(self.fc1(res))
#         h = self.relu(self.fc2(h))
#         # h = self.fc2(h)
#         return h
#
# class MLP_V2S(nn.Module):
#     def __init__(self, opt):
#         super(MLP_V2S, self).__init__()
#         self.individuals = nn.ModuleList([])
#         for i in range(3):
#             self.individuals.append(MLP_V2S_base(opt))
#         self.fc2 = nn.Linear(opt['attSize'], opt['attSize'])
#         self.fc3 = nn.Linear(opt['attSize'], opt['attSize'])
#         self.relu = nn.ReLU(True)
#
#     def forward(self, res):
#         responses = [indiv(res) for indiv in self.individuals]
#         att_anchor = sum(responses)
#         cluster1 = self.relu(self.fc2(att_anchor))
#         cluster2 = self.relu(self.fc3(att_anchor))
#         return att_anchor, cluster1, cluster2


