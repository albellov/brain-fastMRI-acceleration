import os
import argparse
from time import time
from datetime import datetime

import numpy as np
import torch
import torch.utils.data as torch_data

from fastmri.model import UnetGenerator, Discriminator
from fastmri.utils import *


DATA_PREFIX = 'ax_t2_single'

class Pix2PixModel:
    def __init__(self, G, D, G_optimizer, D_optimizer, l1_lambda, acceleration,
                 path_to_results=None, device='cpu', model_name=None):

        self.G = G.to(device)
        self.D = D.to(device)
        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer
        self.l1_lambda = l1_lambda
        self.acceleration = acceleration
        self.device = device

        self.train_losses = {'generator' : [],
                             'discriminator': []
                            }
        self.val_losses =   {'generator' : [],
                             'discriminator': []
                            }

        self.best_discriminator_val_loss = np.inf
        self.best_generator_val_loss = np.inf

        if model_name:
            self.model_name = model_name
        else:
            self.model_name = datetime.now().strftime('%Y-%m-%d-%H.%M.%S')

        if path_to_results:
            self.results_dir = os.path.join(path_to_results, f'x{acceleration}', self.model_name)
            self.log_file = os.path.join(self.results_dir, 'log_' + self.model_name)
            self.losses_file = os.path.join(self.results_dir, 'losses_' + self.model_name + '.pkl')

            os.makedirs(self.results_dir, exist_ok=True)
        else:
            self.log_file = None
            self.results_dir = None
            self.losses_file = None


    def discriminator_loss(self, sampled, source):
        D_result = self.D(sampled, source).squeeze()
        true_answers = torch.ones(D_result.size()).to(self.device)

        D_real_loss = F.binary_cross_entropy(D_result, true_answers)

        generated = self.G(sampled)
        D_result = self.D(sampled, generated).squeeze()
        false_answers = torch.zeros(D_result.size()).to(self.device)

        D_fake_loss = F.binary_cross_entropy(D_result, false_answers)

        loss = (D_real_loss + D_fake_loss) / 2
        return loss

    def train_discriminator(self, sampled, source):
        self.D.zero_grad()

        loss = self.discriminator_loss(sampled, source)

        loss.backward()
        self.D_optimizer.step()

        return loss.data

    def generator_loss(self, sampled, source):
        generated = self.G(sampled)
        D_result = self.D(sampled, generated).squeeze()
        true_answers = torch.ones(D_result.size()).to(self.device)

        loss = F.binary_cross_entropy(D_result, true_answers) + \
               self.l1_lambda*F.l1_loss(generated, source)
        return loss

    def train_generator(self, sampled, source):
        self.G.zero_grad()

        loss = self.generator_loss(sampled, source)

        loss.backward()
        self.G_optimizer.step()

        return loss.data

    def train_step(self, train_loader):
        self.G.train()
        self.D.train()

        discriminator_train_losses = []
        generator_train_losses = []

        for source, sampled in train_loader:

            source = source.to(self.device)
            sampled = sampled.to(self.device)

            discriminator_train_losses.append(self.train_discriminator(sampled, source))
            generator_train_losses.append(self.train_generator(sampled, source))

        return sum(discriminator_train_losses) / len(discriminator_train_losses), \
               sum(generator_train_losses) / len(generator_train_losses)

    def eval(self, val_loader):
        self.G.eval()
        self.D.eval()

        discriminator_val_losses = []
        generator_val_losses = []

        for source, sampled in val_loader:

            source = source.to(self.device)
            sampled = sampled.to(self.device)

            discriminator_val_losses.append(self.discriminator_loss(sampled, source).data)
            generator_val_losses.append(self.generator_loss(sampled, source).data)

        return sum(discriminator_val_losses) / len(discriminator_val_losses), \
               sum(generator_val_losses) / len(generator_val_losses)

    def save_best_D(self):
        last_discriminator_val_loss = self.val_losses['discriminator'][-1]

        if self.best_discriminator_val_loss > last_discriminator_val_loss:

            path_to_model = os.path.join(self.results_dir, f'D_best.pth')
            self.best_discriminator_val_loss = last_discriminator_val_loss
            torch.save(self.D.state_dict(), path_to_model)

    def save_best_G(self):
        last_generator_val_loss = self.val_losses['generator'][-1]

        if self.best_generator_val_loss > last_generator_val_loss:

            path_to_model = os.path.join(self.results_dir, f'G_best.pth')
            self.best_generator_val_loss = last_generator_val_loss
            torch.save(self.G.state_dict(), path_to_model)

    def get_log_line(self, epoch, epochs, epoch_time):

        is_eval = self.val_losses['generator'] and self.val_losses['discriminator']

        last_generator_train_loss = self.train_losses['generator'][-1]
        last_discriminator_train_loss = self.train_losses['discriminator'][-1]

        if is_eval:
            last_generator_val_loss = self.val_losses['generator'][-1]
            last_discriminator_val_loss = self.val_losses['discriminator'][-1]

        log_line = f'Epoch {epoch}/{epochs} | Train loss: [Dis: ' + \
                   f'{last_discriminator_train_loss:.4f}, Gen: {last_generator_train_loss:.4f}]'
        if is_eval:
            log_line += f' | Val loss: [Dis: {last_discriminator_val_loss:.4f}, ' + \
                        f'Gen: {last_generator_val_loss:.4f}]'

        log_line += f' | {epoch_time//60:.0f} min {epoch_time%60:.0f} sec'

        return log_line

    def train(self, epochs, train_loader, val_loader=None, verbose=True, logging=False):
        start_time = time()

        freq = max(epochs // 20, 1)

        for epoch in range(1, epochs+1):
            epoch_start = time()

            discriminator_train_loss, generator_train_loss = self.train_step(train_loader)
            self.train_losses['generator'].append(float(generator_train_loss))
            self.train_losses['discriminator'].append(float(discriminator_train_loss))

            if val_loader:
                discriminator_val_loss, generator_val_loss = self.eval(val_loader)
                self.val_losses['generator'].append(float(generator_val_loss))
                self.val_losses['discriminator'].append(float(discriminator_val_loss))

            if val_loader and self.results_dir:
                self.save_best_D()
                self.save_best_G()

            epoch_time = time() - epoch_start
            log_line = self.get_log_line(epoch, epochs, epoch_time)

            if verbose and epoch%freq == 0:
                print(log_line)

            if logging:
                with open(self.log_file, 'a') as f:
                    f.write(log_line + '\n')

                save_history_loss(self.losses_file, self.train_losses, self.val_losses)

        train_time = time() - start_time
        if logging:
            with open(self.log_file, 'a') as f:
                f.write(f'\nTrain time: {train_time/60:.1f} min\n')
        if verbose:
            print(f'\nTrain time: {train_time/60:.1f} min')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--acceleration', type=int, required=True)
    parser.add_argument('--with_eval', type=int, default=1)
    parser.add_argument('--model_name', type=str, default=None)

    parser.add_argument('--path_to_data', type=str, default='data')
    parser.add_argument('--path_to_results', type=str, default='results')

    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--loader_workers', type=int, default=8)

    parser.add_argument('--random_subset', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--D_lr', type=float, default=0.0002)
    parser.add_argument('--G_lr', type=float, default=0.0002)
    parser.add_argument('--l1_lambda', type=float, default=100)

    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--logging', type=int, default=1)

    return parser.parse_args()


def init_train_loader(args, path_to_source_train, path_to_sampled_train):
    fmri_train = fastMRIData(path_to_source_train, path_to_sampled_train)
    if args.random_subset:
        sampler = torch_data.RandomSampler(fmri_train, replacement=True, num_samples=args.random_subset)
        train_loader = torch_data.DataLoader(fmri_train, sampler=sampler, batch_size=args.train_batch_size,
                                             shuffle=False, num_workers=args.loader_workers)
    else:
        train_loader = torch_data.DataLoader(fmri_train, batch_size=args.train_batch_size,
                                             shuffle=True, num_workers=args.loader_workers)

    return train_loader


def init_val_loader(args, path_to_source_val, path_to_sampled_val):
    if args.with_eval:
        fmri_val = fastMRIData(path_to_source_val, path_to_sampled_val)

        val_loader = torch_data.DataLoader(fmri_val, batch_size=args.val_batch_size,
                                           shuffle=False, num_workers=args.loader_workers)
    else:
        val_loader = None

    return val_loader


def init_data_loaders(args):
    torch.manual_seed(args.random_state)

    path_to_source_train = os.path.join(args.path_to_data, f'{DATA_PREFIX}_source_train')
    path_to_source_val = os.path.join(args.path_to_data, f'{DATA_PREFIX}_source_val')
    path_to_sampled_train = os.path.join(args.path_to_data,
                                         f'{DATA_PREFIX}_sampled_x{args.acceleration}_train')
    path_to_sampled_val = os.path.join(args.path_to_data,
                                       f'{DATA_PREFIX}_sampled_x{args.acceleration}_val')

    train_loader = init_train_loader(args, path_to_source_train, path_to_sampled_train)
    val_loader = init_val_loader(args, path_to_source_val, path_to_sampled_val)

    if args.verbose:
        print(f'Train data: {path_to_source_train} | {path_to_sampled_train}')
        print(f'Val data: {path_to_source_val} | {path_to_sampled_val}')

    return train_loader, val_loader


def main():
    args = parse_args()

    torch.manual_seed(args.random_state)

    G = UnetGenerator()
    D = Discriminator()

    betas = (0.5, 0.999)

    G_optimizer = torch.optim.Adam(G.parameters(), lr=args.G_lr, betas=betas)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=args.D_lr, betas=betas)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if device == torch.device('cuda:0'):
        torch.cuda.empty_cache()

    train_loader, val_loader = init_data_loaders(args)

    pix2pix = Pix2PixModel(G, D, G_optimizer, D_optimizer, args.l1_lambda, args.acceleration,
                           args.path_to_results, device, args.model_name)

    if args.verbose:
        print('Path to results:', pix2pix.results_dir)
        print('Device:', device)
        print('Model_name:', pix2pix.model_name)
        print()

    if args.logging:
        write_train_params(pix2pix.log_file, device=device, random_state=args.random_state,
                           epochs=args.epochs, l1_lambda=args.l1_lambda,
                           G_optimizer=G_optimizer, D_optimizer=D_optimizer)

    pix2pix.train(args.epochs, train_loader, val_loader,
                  verbose=args.verbose, logging=args.logging)


if __name__ == '__main__':
    main()
