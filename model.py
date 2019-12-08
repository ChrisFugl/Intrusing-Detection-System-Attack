import os
import numpy as np
from scores import get_binary_class_scores
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

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
      *block(input_size, 256),
      *block(256, 256),
      *block(256, 256),
      *block(256, 256),
      nn.Linear(256, output_size),
      nn.ReLU()
    )

  def forward(self, x):
    adv_traffic = self.model(x)
    return adv_traffic

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
      *block(input_size, 256),
      *block(256, 256),
      *block(256, 256),
      *block(256, 256),
      nn.Linear(256, 1),
      nn.Sigmoid()
    )

  def forward(self, x):
    traffic = self.model(x)
    return traffic

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

    self.writer_train = SummaryWriter(log_dir=f'runs/{options.name}/train')
    self.writer_val = SummaryWriter(log_dir=f'runs/{options.name}/val')

  def train(self, trainingset, validationset):
    self.generator.train()
    self.discriminator.train()

    normal_traffic, nff_traffic, normal_labels, nff_labels = trainingset
    normal_traffic_val, nff_traffic_val, normal_labels_val, nff_labels_val = validationset
    normal_traffic_val_tensor = torch.Tensor(normal_traffic_val)

    n_observations_mal = len(nff_traffic)
    total_mal_batches = n_observations_mal // self.batch_size

    n_observations_nor = len(normal_traffic)
    total_nor_batches = n_observations_nor // self.batch_size

    iterations = 0
    
    for epoch in range(self.max_epoch):

      for batch_number in range(total_nor_batches):
        run_loss_g = 0.
        run_loss_d = 0.

        batch_start = batch_number * self.batch_size
        batch_finish = (batch_number + 1) * self.batch_size
        batch = torch.from_numpy(normal_traffic[batch_start:batch_finish]).float() # 64*23 for DoS
        batch = Variable(batch.to(self.device))

        # Discriminator training
        for c in range(self.critic_iter):
          # With Adversarial data
          batch_Malicious = Variable(torch.Tensor(nff_traffic[np.random.randint(0, n_observations_mal, self.batch_size)])) # 64*23
          noise = Variable(torch.rand(self.batch_size, self.noise_dim)) # 64*9
          batch_Malicious_noise = Variable(torch.cat((batch_Malicious, noise), 1)) # 64*32
          batch_Malicious_noise = Variable(batch_Malicious_noise.to(self.device))

          adv_traffic = self.generator(batch_Malicious_noise)

          # With real data
          loss_d_real = torch.mean(self.discriminator(batch))

          # With Adversarial data
          loss_d_adv = torch.mean(self.discriminator(adv_traffic))

          loss_d = loss_d_adv - loss_d_real

          self.optim_D.zero_grad()
          loss_d.backward()
          self.optim_D.step()

          run_loss_d += loss_d.item()

          for p in self.discriminator.parameters():
            p.data.clamp_(-self.weight_clipping, self.weight_clipping)

        # Generator training
        batch_Malicious = Variable(torch.Tensor(nff_traffic[np.random.randint(0, n_observations_mal, self.batch_size)])) # 64*23
        noise = Variable(torch.rand(self.batch_size, self.noise_dim)) # 64*9
        batch_Malicious_noise = Variable(torch.cat((batch_Malicious, noise), 1)) # 64*32
        batch_Malicious_noise = Variable(batch_Malicious_noise.to(self.device))

        adv_traffic = self.generator(batch_Malicious_noise) # 64*23

        loss_g = -torch.mean(self.discriminator(adv_traffic))

        self.optim_G.zero_grad()
        loss_g.backward()
        self.optim_G.step()

        run_loss_g += loss_g.item()

        loss_d_avg = run_loss_d / self.critic_iter
        loss_g_avg = run_loss_g

        self.writer_train.add_scalar('loss/d', loss_d_avg, iterations)
        self.writer_train.add_scalar('loss/g', loss_g_avg, iterations)
        iterations += 1
      
      self.generator.eval()
      self.discriminator.eval()
      
      malicious_val = Variable(torch.Tensor(nff_traffic_val))
      noise_val = Variable(torch.rand(len(nff_labels_val), self.noise_dim))
      malicious_noise_val = Variable(torch.cat((malicious_val, noise_val), 1))
      malicious_noise_val = Variable(malicious_noise_val.to(self.device))
      adv_traffic_val = self.generator(malicious_noise_val).cpu().detach()

      loss_d_real_val = torch.mean(self.discriminator(normal_traffic_val_tensor))
      loss_d_adv_val = torch.mean(self.discriminator(adv_traffic_val))
      loss_d_val = loss_d_adv_val - loss_d_real_val
      loss_g_val = -torch.mean(self.discriminator(adv_traffic_val))
      self.writer_val.add_scalar('loss/d', loss_d_val, iterations)
      self.writer_val.add_scalar('loss/g', loss_g_val, iterations)

      predictions_adv_val = self.predict(adv_traffic_val)
      accuracy_val, f1_val, precision_val, recall_val = get_binary_class_scores(nff_labels_val, predictions_adv_val)
      self.writer_val.add_scalar('adv_scores_val/accuracy', accuracy_val, iterations)
      self.writer_val.add_scalar('adv_scores_val/f1', f1_val, iterations)
      self.writer_val.add_scalar('adv_scores_val/precision', precision_val, iterations)
      self.writer_val.add_scalar('adv_scores_val/recall', recall_val, iterations)

      predictions_normal_val = self.predict(normal_traffic_val_tensor)
      accuracy_val, f1_val, precision_val, recall_val = get_binary_class_scores(normal_labels_val, predictions_normal_val)
      self.writer_val.add_scalar('normal_scores_val/accuracy', accuracy_val, iterations)
      self.writer_val.add_scalar('normal_scores_val/f1', f1_val, iterations)
      self.writer_val.add_scalar('normal_scores_val/precision', precision_val, iterations)
      self.writer_val.add_scalar('normal_scores_val/recall', recall_val, iterations)

      self.generator.train()
      self.discriminator.train()

      print(
        "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, self.max_epoch, loss_d_val, loss_g_val)
      )
     
  def predict(self, traffic):
    outputs = self.discriminator(traffic).detach()
    predictions = torch.empty_like(outputs)
    predictions[outputs < 0.5] = 0
    predictions[outputs >= 0.5] = 1
    return predictions.cpu().numpy()
  
  def predict_normal_and_adversarial(self, normal_traffic, malicious_traffic):
    self.generator.eval()
    self.discriminator.eval()

    n_observations_mal = len(malicious_traffic)

    batch_Malicious = torch.from_numpy(malicious_traffic).float() 
    noise = Variable(torch.rand(n_observations_mal, self.noise_dim))
    batch_Malicious_noise = torch.cat((batch_Malicious, noise), 1) 
    batch_Malicious_noise = Variable(batch_Malicious_noise.to(self.device))

    adversarial = self.generator(batch_Malicious_noise)
    input_disc = Variable(torch.Tensor(np.concatenate((normal_traffic, adversarial.cpu().detach().numpy()), axis=0)))
    input_disc = Variable(input_disc.to(self.device))
    outputs = self.discriminator(input_disc)

    predictions = torch.empty_like(outputs)
    predictions[outputs < 0.5] = 0
    predictions[outputs >= 0.5] = 1
    return predictions.cpu().numpy()

  def generate(self, malicious_traffic, type):
    self.generator.eval()
    self.discriminator.eval()

    n_observation_mal = len(malicious_traffic)

    batch_Malicious = torch.from_numpy(malicious_traffic).float()
    noise = Variable(torch.rand(n_observation_mal, self.noise_dim))
    batch_Malicious_noise = torch.cat((batch_Malicious, noise), 1) 
    batch_Malicious_noise = Variable(batch_Malicious_noise.to(self.device))

    adversarial = self.generator(batch_Malicious_noise)

  def save(self, path):
    if not os.path.exists(path):
      os.makedirs(path)
    torch.save(self.generator.state_dict(), path + 'generator.pt')
    torch.save(self.discriminator.state_dict(), path + 'discriminator.pt')

  def load(self, path):
    self.generator.load_state_dict(torch.load(path + 'generator.pt'))
    self.discriminator.load_state_dict(torch.load(path + 'discriminator.pt'))
