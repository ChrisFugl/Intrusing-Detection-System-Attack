import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class Generator(nn.Module):
  def __init__(self, input_size, output_size):
    super(Generator, self).__init__()

    def block(input_dim, output_dim, normalize=True):
      layers = [nn.Linear(input_dim, output_dim)]
      #if normalize:
      #  layers.append(nn.BatchNorm1d(output_dim))
      layers.append(nn.ReLU(inplace=True))
      return layers

    self.model = nn.Sequential(
      *block(input_size, input_size//2),
      *block(input_size//2, input_size//2),
      *block(input_size//2, input_size//2),
      *block(input_size//2, input_size//2),
      nn.Linear(input_size//2, output_size)
    )

  def forward(self, x):
    adv_traffic = self.model(x)
    return adv_traffic

class discriminatorOutput(nn.Module):
  def __init__(self):
    super(discriminatorOutput, self).__init__()

    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    output = (1 + self.sigmoid(x)) / 2
    return output

class Discriminator(nn.Module):
  def __init__(self, input_size):
    super(Discriminator, self).__init__()

    def block(input_dim, output_dim, normalize=True):
      layers = [nn.Linear(input_dim, output_dim)]
      #if normalize:
      #  layers.append(nn.BatchNorm1d(output_dim))
      layers.append(nn.LeakyReLU(inplace=True))
      return layers

    self.model = nn.Sequential(
      *block(input_size, input_size*2),
      *block(input_size*2, input_size*2),
      *block(input_size*2, input_size*2),
      *block(input_size*2, input_size//2),
      nn.Linear(input_size//2, 1)
    )

    self.output = discriminatorOutput()

  def forward(self, x):
    traffic = self.model(x)
    output = self.output(traffic)
    return output

class WGAN(object):
  def __init__(self, options, n_attributes):
    self.n_attributes = n_attributes
    self.noise_dim = options.noise_dim
    self.generator = Generator(self.n_attributes + self.noise_dim, self.n_attributes)
    self.discriminator = Discriminator(self.n_attributes)

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.generator.to(self.device)
    self.discriminator.to(self.device)

    self.max_epoch = options.epochs
    self.batch_size = options.batch_size
    self.learning_rate = options.learning_rate
    self.weight_clipping = options.weight_clipping
    self.critic_iter = options.critic_iter

    self.optim_G = optim.RMSprop(self.generator.parameters(), self.learning_rate)
    self.optim_D = optim.RMSprop(self.discriminator.parameters(), self.learning_rate)

  def train(self, normal_traffic, nff_traffic, normal_labels, nff_labels):
    self.generator.train()
    self.discriminator.train()

    n_observations_mal = len(nff_traffic)
    total_mal_batches = n_observations_mal // self.batch_size

    n_observations_nor = len(normal_traffic)
    total_nor_batches = n_observations_nor // self.batch_size
    
    for epoch in range(self.max_epoch):

      run_loss_g = 0.
      run_loss_d = 0.

      for batch_number in range(total_nor_batches):
        batch_start = batch_number * self.batch_size
        batch_finish = (batch_number + 1) * self.batch_size
        batch = torch.from_numpy(normal_traffic[batch_start:batch_finish]).float() # 64*23 for DoS

        # Discriminator training
        for c in range(self.critic_iter):
          # With real data
          self.optim_D.zero_grad()

          batch = Variable(batch.to(self.device))
          loss_d_real = torch.mean(self.discriminator(batch))
          
          # With Adversarial data
          batch_Malicious = Variable(torch.Tensor(nff_traffic[np.random.randint(0, n_observations_mal, self.batch_size)])) # 64*23
          noise = Variable(torch.randn(self.batch_size, self.noise_dim)) # 64*9
          batch_Malicious_noise = Variable(torch.cat((batch_Malicious, noise), 1)) # 64*32
          batch_Malicious_noise = Variable(batch_Malicious_noise.to(self.device))
          
          adv_traffic = self.generator(batch_Malicious_noise)
          loss_d_adv = torch.mean(self.discriminator(adv_traffic))

          loss_d = loss_d_adv - loss_d_real
          loss_d.backward()
          self.optim_D.step()

          run_loss_d += loss_d.item()

          for p in self.discriminator.parameters():
            p.data.clamp_(-self.weight_clipping, self.weight_clipping)

        # Generator training
        self.optim_G.zero_grad()

        batch_Malicious = Variable(torch.Tensor(nff_traffic[np.random.randint(0, n_observations_mal, self.batch_size)])) # 64*23
        noise = Variable(torch.randn(self.batch_size, self.noise_dim)) # 64*9
        batch_Malicious_noise = Variable(torch.cat((batch_Malicious, noise), 1)) # 64*32
        batch_Malicious_noise = Variable(batch_Malicious_noise.to(self.device))
        
        adv_traffic = self.generator(batch_Malicious_noise) # 64*23
        
        loss_g = -torch.mean(self.discriminator(adv_traffic))
        loss_g.backward()

        self.optim_G.step()

        run_loss_g += loss_g.item()

        

      print(
        "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, self.max_epoch, run_loss_d/self.critic_iter, run_loss_g)
      )
     
  
  def predict(self, normal_traffic, malicious_traffic):
    self.generator.eval()
    self.discriminator.eval()

    batch_Malicious = torch.from_numpy(malicious_traffic).float() # 64*23
    noise = Variable(torch.randn(self.batch_size, self.noise_dim)) # 64*9
    batch_Malicious_noise = torch.cat((batch_Malicious, noise), 1) # 64*32
    batch_Malicious_noise = Variable(batch_Malicious_noise.to(self.device))


  def save(self, path):
    if not os.path.exists(path):
      os.makedirs(path)
    torch.save(self.generator.state_dict(), path + 'generator.pt')
    torch.save(self.discriminator.state_dict(), path + 'discriminator.pt')

  def load(self, path):
    self.generator.load_state_dict(torch.load(path + 'generator.pt'))
    self.discriminator.load_state_dict(torch.load(path + 'discriminator.pt'))
