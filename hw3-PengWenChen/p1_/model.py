import torch.nn as nn
import torch
from torch.autograd import Variable
import pdb
class VAE(nn.Module):
    def __init__(self, latent_dim=1024):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            self.encoder_block(3, 32),
            self.encoder_block(32, 64),
            self.encoder_block(64, 128),
            self.encoder_block(128, 256),
            self.encoder_block(256, 512),
        )
        self.fc_mu = nn.Linear(512*4, self.latent_dim)
        self.fc_var = nn.Linear(512*4, self.latent_dim)

        self.decode_input = nn.Linear(self.latent_dim, 512*4)

        self.decoder = nn.Sequential(
            self.decoder_block(512, 256),
            self.decoder_block(256, 128),
            self.decoder_block(128, 64),
            self.decoder_block(64, 32),
            self.decoder_block(32, 32),
        )

        self.final_layer = nn.Sequential(
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh(),
            # nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.encoder(input)
        x = torch.flatten(x, start_dim=1)
        # pdb.set_trace()
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        z = self.sample_z(mu, log_var)

        recon = self.decode_input(z)
        recon = recon.view(-1, 512, 2, 2)
        # recon = recon.view(-1, 256, 4, 4)
        recon = self.decoder(recon)
        recon = self.final_layer(recon)

        return mu, log_var, recon
    
    def encoder_block(self, input_ch, output_ch):
        return nn.Sequential(
            nn.Conv2d(input_ch, output_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(output_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )
    
    def decoder_block(self, input_ch, output_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(input_ch, output_ch, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(output_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )
    
    def sample_z(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu


# class VAE(nn.Module):
#     def __init__(self, latent_size):
#         super(VAE, self).__init__()
#         # torch.nn.Conv2d(in_channels, out_channels, kernel_size, 
#         # stride=1, padding=0, dilation=1, groups=1, bias=True)
#         self.latent_size = latent_size
#         self.conv_stage = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=(4, 4), stride=(2, 2), padding=1),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.01, inplace=True),
            
#             nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.01, inplace=True),
            
#             nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.01, inplace=True),
            
#             nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.01, inplace=True),
#         )
#         self.fcMean = nn.Linear(4096, self.latent_size)
#         self.fcStd = nn.Linear(4096, self.latent_size)
        
#         self.fcDecode = nn.Linear(self.latent_size,4096)
        
#         self.trans_conv_stage = nn.Sequential(

#             nn.ConvTranspose2d(256, 128, 4, 2, padding=1,bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.01, inplace=True),
 
#             nn.ConvTranspose2d(128, 64, 4, 2, padding=1,bias=False),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.01, inplace=True),
 
#             nn.ConvTranspose2d(64, 32, 4, 2, padding=1,bias=False),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.01, inplace=True),
            
#             nn.ConvTranspose2d(32, 3, 4, 2, padding=1,bias=False)
#         )
#         # final output activation function
#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()
#     def encode(self, x):
#         conv_output = self.conv_stage(x).view(-1, 4096)
#         return self.fcMean(conv_output), self.fcStd(conv_output)

#     def reparameterize(self, mu, logvar):
#         std = logvar.mul(0.5).exp_()
#         eps = torch.FloatTensor(std.size()).normal_()
#         eps = Variable(eps).cuda()

#         return eps.mul(std).add_(mu)


#     def decode(self, z):
#         fc_output = self.fcDecode(z).view(-1, 256, 4, 4)
# #         print("decode fc output", fc_output.size())
#         trans_conv_output = self.trans_conv_stage(fc_output)
# #         print("trans_conv_output", trans_conv_output.size())
#         return self.tanh(trans_conv_output)

#     def forward(self, x):
#         mu, logvar = self.encode(x)
# #         print("mu shape",mu.size()," logvar",logvar.size())
#         z = self.reparameterize(mu, logvar)
# #         print("z shape",z.shape)
#         return self.decode(z), mu, logvar
