import itertools
import os
import numpy as np
from scores import get_binary_class_scores
import time
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
    adversarial_traffic = self.model(x)
    return adversarial_traffic

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

    self.epochs = options.epochs
    self.batch_size = options.batch_size
    self.learning_rate = options.learning_rate
    self.weight_clipping = options.weight_clipping
    self.critic_iter = options.critic_iter

    self.optim_G = optim.RMSprop(self.generator.parameters(), self.learning_rate)
    self.optim_D = optim.RMSprop(self.discriminator.parameters(), self.learning_rate)

    self.one = torch.FloatTensor([1.0], device=self.device)
    self.negative_one = self.one * -1

    self.writer_train = SummaryWriter(log_dir=f'runs/{options.name}/train')
    self.writer_val = SummaryWriter(log_dir=f'runs/{options.name}/val')

    self.start_epoch = 0
    self.start_iteration = 0

    self.checkpoint_directory = os.path.join(options.checkpoint_directory, options.name)
    self.checkpoint_interval_s = options.checkpoint_interval_s
    os.makedirs(self.checkpoint_directory, exist_ok=True)
    self.previous_checkpoint_time = time.time()
    if options.checkpoint is not None:
        self.load_checkpoint(options.checkpoint)

  def train(self, trainingset, validationset):
    self.generator.train()
    self.discriminator.train()

    normal_traffic, nff_traffic, normal_labels, nff_labels = trainingset
    normal_labels = 1 - normal_labels
    nff_labels = 1 - nff_labels
    malicious_traffic_tensor = torch.Tensor(nff_traffic)
    normal_traffic_tensor = torch.Tensor(normal_traffic)

    normal_traffic_val, nff_traffic_val, normal_labels_val, nff_labels_val = validationset
    normal_labels_val = 1 - normal_labels_val
    nff_labels_val = 1 - nff_labels_val
    malicious_traffic_val_tensor = torch.Tensor(nff_traffic_val)
    normal_traffic_val_tensor = torch.Tensor(normal_traffic_val)

    n_observations_mal = len(nff_traffic)

    n_observations_nor = len(normal_traffic)
    total_nor_batches = n_observations_nor // self.batch_size

    iterations = self.start_iteration

    if self.epochs < 0:
        epoch_iterator = itertools.count(self.start_epoch)
    else:
        epoch_iterator = range(self.start_epoch, self.epochs)

    for epoch in epoch_iterator:
      for batch_number in range(total_nor_batches):
        discriminated_adversarial_sum = 0.0
        discriminated_normal_sum = 0.0

        batch_start = batch_number * self.batch_size
        batch_finish = (batch_number + 1) * self.batch_size
        batch = torch.from_numpy(normal_traffic[batch_start:batch_finish]).float() # 64*23 for DoS
        batch = Variable(batch.to(self.device))

        for parameter in self.discriminator.parameters():
            parameter.requires_grad = True
        for parameter in self.generator.parameters():
            parameter.requires_grad = False

        # Discriminator training
        for c in range(self.critic_iter):
          # With Adversarial data
          batch_Malicious = torch.Tensor(nff_traffic[np.random.randint(0, n_observations_mal, self.batch_size)]) # 64*23
          noise = torch.rand(self.batch_size, self.noise_dim) # 64*9
          batch_Malicious_noise = torch.cat((batch_Malicious, noise), 1) # 64*32
          batch_Malicious_noise = Variable(batch_Malicious_noise.to(self.device))

          adversarial_traffic = self.generator(batch_Malicious_noise)

          # With real data
          discriminated_normal = torch.mean(self.discriminator(batch)).view(1)

          # With Adversarial data
          discriminated_adversarial = torch.mean(self.discriminator(adversarial_traffic)).view(1)

          discriminator_loss = - (discriminated_normal - discriminated_adversarial)

          self.optim_D.zero_grad()
          #discriminated_normal.backward(self.one)
          #discriminated_adversarial.backward(self.negative_one)
          discriminator_loss.backward()
          self.optim_D.step()

          discriminated_adversarial_sum += discriminated_adversarial.item()
          discriminated_normal_sum += discriminated_normal.item()

          for p in self.discriminator.parameters():
            p.data.clamp_(-self.weight_clipping, self.weight_clipping)

        # Generator training
        for parameter in self.discriminator.parameters():
            parameter.requires_grad = False
        for parameter in self.generator.parameters():
            parameter.requires_grad = True

        batch_Malicious = torch.Tensor(nff_traffic[np.random.randint(0, n_observations_mal, self.batch_size)]) # 64*23
        noise = torch.rand(self.batch_size, self.noise_dim) # 64*9
        batch_Malicious_noise = torch.cat((batch_Malicious, noise), 1) # 64*32
        batch_Malicious_noise = Variable(batch_Malicious_noise.to(self.device))

        adversarial_traffic = self.generator(batch_Malicious_noise) # 64*23

        generator_objective = torch.mean(self.discriminator(adversarial_traffic)).view(1)
        generator_loss = - generator_objective

        self.optim_G.zero_grad()
        generator_loss.backward()
        self.optim_G.step()

        discriminated_adversarial_avg = discriminated_adversarial_sum / self.critic_iter
        discriminated_normal_avg = discriminated_normal_sum / self.critic_iter
        discriminator_objective_avg = discriminated_normal_avg - discriminated_adversarial_avg

        self.writer_train.add_scalar('discriminator/adversarial_mean', discriminated_adversarial_avg, iterations)
        self.writer_train.add_scalar('discriminator/normal_mean', discriminated_normal_avg, iterations)
        self.writer_train.add_scalar('discriminator/objective', discriminator_objective_avg, iterations)
        self.writer_train.add_scalar('generator/objective', generator_objective.item(), iterations)

        iterations += 1

      self.generator.eval()
      self.discriminator.eval()
      self.log_stats_to_tensorboard(
        self.writer_train,
        iterations,
        normal_traffic_tensor,
        normal_labels,
        malicious_traffic_tensor,
        nff_labels
      )
      discriminated_adversarial_mean_val, discriminated_normal_mean_val, discriminator_objective_val, generator_objective_val = self.log_stats_to_tensorboard(
        self.writer_val,
        iterations,
        normal_traffic_val_tensor,
        normal_labels_val,
        malicious_traffic_val_tensor,
        nff_labels_val
      )
      self.writer_val.add_scalar('discriminator/adversarial_mean', discriminated_adversarial_mean_val, iterations)
      self.writer_val.add_scalar('discriminator/normal_mean', discriminated_normal_mean_val, iterations)
      self.writer_val.add_scalar('discriminator/objective', discriminator_objective_val, iterations)
      self.writer_train.add_scalar('generator/objective', generator_objective_val, iterations)

      self.generator.train()
      self.discriminator.train()

      print(
        "[Epoch %d/%d] [D means normal: %f] [D means adversarial: %f] {D objective: %f}"
        % (epoch, self.epochs, discriminated_normal_mean_val, discriminated_adversarial_mean_val, discriminator_objective_val)
      )

      current_time = time.time()
      if self.checkpoint_interval_s <= current_time - self.previous_checkpoint_time:
        self.save_checkpoint(epoch + 1, iterations)
        self.previous_checkpoint_time = time.time()

  def log_stats_to_tensorboard(self, writer, iterations, normal_traffic, normal_labels, malicious_traffic, malicious_labels):
      noise = torch.rand(len(malicious_traffic), self.noise_dim)
      malicious_noise = torch.cat((malicious_traffic, noise), 1).to(self.device)
      adversarial_traffic = self.generator(malicious_noise)

      discriminated_adversarial_mean = torch.mean(self.discriminator(adversarial_traffic))
      discriminated_normal_mean = torch.mean(self.discriminator(normal_traffic))
      discriminator_objective = discriminated_normal_mean - discriminated_adversarial_mean
      generator_objective = discriminated_adversarial_mean

      predictions_adversarial = self.predict(adversarial_traffic)
      accuracy, f1, precision, recall, _ = get_binary_class_scores(malicious_labels, predictions_adversarial)
      writer.add_scalar('scores_adversarial/accuracy', accuracy, iterations)
      writer.add_scalar('scores_adversarial/f1', f1, iterations)
      writer.add_scalar('scores_adversarial/precision', precision, iterations)
      writer.add_scalar('scores_adversarial/recall', recall, iterations)

      predictions_normal = self.predict(normal_traffic)
      accuracy, f1, precision, recall, _ = get_binary_class_scores(normal_labels, predictions_normal)
      writer.add_scalar('scores_normal/accuracy', accuracy, iterations)
      writer.add_scalar('scores_normal/f1', f1, iterations)
      writer.add_scalar('scores_normal/precision', precision, iterations)
      writer.add_scalar('scores_normal/recall', recall, iterations)

      return discriminated_adversarial_mean, discriminated_normal_mean, discriminator_objective, generator_objective

  def predict(self, traffic):
    outputs = self.discriminator(traffic).squeeze()
    predictions = torch.empty((len(outputs),), dtype=torch.uint8)
    predictions[outputs < 0.5] = 0   # adversarial traffic
    predictions[outputs >= 0.5] = 1  # normal traffic
    return predictions.numpy()

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
    predictions[outputs < 0.5] =  0
    predictions[outputs >= 0.5] = 1
    return predictions.cpu().numpy()

  def generate(self, malicious_traffic):
    self.generator.eval()
    self.discriminator.eval()

    n_observations_mal = len(malicious_traffic)

    batch_Malicious = torch.Tensor(malicious_traffic)
    noise = torch.rand(n_observations_mal, self.noise_dim)
    batch_Malicious_noise = torch.cat((batch_Malicious, noise), 1)
    batch_Malicious_noise = Variable(batch_Malicious_noise.to(self.device))

    adversarial = self.generator(batch_Malicious_noise)

    return adversarial

  def save(self, path):
    if not os.path.exists(path):
      os.makedirs(path)
    torch.save(self.generator.state_dict(), path + 'generator.pt')
    torch.save(self.discriminator.state_dict(), path + 'discriminator.pt')

  def load(self, path):
    self.generator.load_state_dict(torch.load(path + 'generator.pt'))
    self.discriminator.load_state_dict(torch.load(path + 'discriminator.pt'))

  def load_checkpoint(self, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    self.generator.load_state_dict(checkpoint['generator'])
    self.discriminator.load_state_dict(checkpoint['discriminator'])
    self.optim_G.load_state_dict(checkpoint['generator_optimizer'])
    self.optim_D.load_state_dict(checkpoint['discriminator_optimizer'])
    self.start_epoch = checkpoint['epoch']
    self.start_iteration = checkpoint['iteration']

  def save_checkpoint(self, epoch, iteration):
    checkpoint = {
      'generator': self.generator.state_dict(),
      'discriminator': self.discriminator.state_dict(),
      'generator_optimizer': self.optim_G.state_dict(),
      'discriminator_optimizer': self.optim_D.state_dict(),
      'epoch': epoch,
      'iteration': iteration,
    }
    checkpoint_path = os.path.join(self.checkpoint_directory, f'epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
