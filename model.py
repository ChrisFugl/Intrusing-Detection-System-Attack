import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class Generator(nn.Module):
  def __init__(self, input_size):
    super(Generator, self).__init__()

    def block(input_dim, output_dim, normalize=True):
      layers = [nn.Linear(input_dim, output_dim)]
      if normalize:
        layers.append(nn.BatchNorm1d(output_dim))
      layers.append(nn.ReLU(inplace=True))
      return layers

    self.model = nn.Sequential(
      *block(input_size, 1024),
      *block(1024, 512),
      *block(512, 256),
      *block(256, 128),
      nn.Linear(128, input_size),
      nn.Tanh()
    )

  def forward(self, x):
    adv_traffic = self.model(x)
    return adv_traffic

class Discriminator(nn.Module):
  def __init__(self, input_size):
    super(Discriminator, self).__init__()

    def block(input_dim, output_dim, normalize=True):
      layers = [nn.Linear(input_dim, output_dim)]
      if normalize:
        layers.append(nn.BatchNorm1d(output_dim))
      layers.append(nn.LeakyReLU(0.2, inplace=True))
      return layers

    self.model = nn.Sequential(
      *block(input_size, 256),
      *block(256, 512),
      *block(512, 1024),
      nn.Linear(1024, 1), 
    )

  def forward(self, x):
    traffic = self.model(x)
    return traffic

class WGAN(object):
  def __init__(self, options, n_attributes):
    self.n_attributes = n_attributes
    self.generator = Generator(n_attributes)
    self.discriminator = Discriminator(n_attributes)

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.generator.to(self.device)
    self.discriminator.to(self.device)

    self.max_epoch = options.epochs
    self.batch_size = options.batch_size
    self.learning_rate = options.learning_rate
    self.weight_clipping = options.weight_clipping
    self.noise_dim = options.noise_dim
    self.critic_iter = options.critic_iter

    self.optim_G = optim.RMSprop(self.generator.parameters(), self.learning_rate)
    self.optim_D = optim.RMSprop(self.discriminator.parameters(), self.learning_rate)

  def train(self, normal_traffic, malicious_traffic):
    self.generator.train()
    self.discriminator.train()

    n_observations_mal = len(malicious_traffic)
    total_mal_batches = n_observations_mal // self.batch_size
    
    for epoch in range(self.max_epoch):
      input_discriminator = normal_traffic

      # Generator training
      for batch_number in range(total_mal_batches):
        batch_start = batch_number * self.batch_size
        batch_finish = (batch_number + 1) * self.batch_size
        batch_Malicious = torch.from_numpy(malicious_traffic[batch_start:batch_finish]).float() # 64*123

        self.optim_G.zero_grad()

        noise = Variable(torch.randn(self.noise_dim, self.n_attributes))
        batch_Malicious_noise = torch.cat((batch_Malicious, noise), 0) # 73*123
        batch_Malicious_noise = Variable(batch_Malicious_noise.to(self.device))

        adv_traffic = self.generator(batch_Malicious_noise) # 73*123

        loss_g = self.discriminator(adv_traffic)
        loss_g = loss_g.mean(0) 
        loss_g.backward()
        cost_g = -loss_g

        self.optim_G.step()

        input_discriminator = np.concatenate((input_discriminator, adv_traffic.cpu().detach().numpy()), axis=0)

      # Discriminator training
      n_observations = len(input_discriminator)
      total_batches = n_observations // self.batch_size

      for batch_number in range(total_batches):
        batch_start = batch_number * self.batch_size
        batch_finish = (batch_number + 1) * self.batch_size
        batch = torch.from_numpy(input_discriminator[batch_start:batch_finish]).float() # 64*123

        self.optim_D.zero_grad()

        for p in self.discriminator.parameters():
            p.data.clamp_(-self.weight_clipping, self.weight_clipping)

        batch = Variable(batch.to(self.device))
        loss_d = -torch.mean(self.discriminator(batch))
        loss_d.backward()

        self.optim_G.step()

      print(
        "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, self.max_epoch, loss_d.item()*100, cost_g.item()*100)
      )
     
  
  def predict(self, malicious_traffic, labels):
    self.generator.eval()
    self.discriminator.eval()

    batch = torch.from_numpy(malicious_traffic).float()
    noise = Variable(torch.randn(self.noise_dim, self.n_attributes))
    batch_noise = torch.cat((batch, noise), 0)
    batch_noise = Variable(batch_noise.to(self.device))
    labels = np.append(labels, [ True for i in range(self.noise_dim) ]).astype(np.float)

    samples = self.generator(batch_noise)
    outputs = self.discriminator(samples)
    predictions = torch.empty_like(outputs)
    predictions[outputs < 0] = 0
    predictions[outputs >= 0] = 1
    return predictions.cpu().detach().numpy(), labels

  def save(self, path):
    if not os.path.exists(path):
      os.makedirs(path)
    torch.save(self.generator.state_dict(), path + 'generator.pt')
    torch.save(self.discriminator.state_dict(), path + 'discriminator.pt')

  def load(self, path):
    self.generator.load_state_dict(torch.load(path + 'generator.pt'))
    self.discriminator.load_state_dict(torch.load(path + 'discriminator.pt'))