import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch.optim as optim
from torch.utils.data import DataLoader


from Models.actor import DRL4TSP
from Tasks import vrp
from Tasks.vrp import VehicleRoutingDataset
from Tasks.gpu_asg import GpuAssignmentDataset
from Models.critc import StateCritic
from util.logger import Logger

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled=False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Detected device {}'.format(device))


def validate(logger, epoch, data_loader, actor, reward_fn, render_fn=None, save_dir='.',
             num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""
    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)

        x0 = x0.to(device) if len(x0) > 0 else None
        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)
        reward = reward_fn(static, tour_indices).mean().item()
        rewards.append(reward)

        if render_fn is not None and batch_idx < num_plot:
            name = 'batch%d_%2.4f'%(batch_idx, reward)
            render_fn(logger, dynamic, name, epoch, tour_indices)

    actor.train()
    return np.mean(rewards)

def train(actor, critic, task, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm, num_epoch,
          **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""

    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    save_dir = os.path.join(task, '%d' % num_nodes, now)

    print('Starting training')

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)
    logger = Logger(f'./logs/train')

    best_reward = np.inf

    for epoch in range(num_epoch):
        print(f'epoch: {epoch}')
        actor.train()
        critic.train()

        times, losses, rewards, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start = epoch_start
        for batch_idx, batch in enumerate(train_loader):

            static, dynamic, x0 = batch

            static = static.to(device)
            dynamic = dynamic.to(device)
            x0 = x0.to(device) if len(x0) > 0 else None
            # Full forward pass through the dataset
            tour_indices, tour_logp = actor(static, dynamic, x0)
            # Sum the log probabilities for each city in the tour


            reward = reward_fn(static, tour_indices)

            # Query the critic for an estimate of the reward
            critic_est = critic(static, dynamic).view(-1)

            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            critic_rewards.append(torch.mean(critic_est.detach()).item())
            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())

            if (batch_idx + 1) % 100 == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-100:])
                mean_reward = np.mean(rewards[-100:])

                print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(train_loader), mean_reward, mean_loss,
                       times[-1]))

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)

        # Save rendering of validation set tours
        valid_dir = os.path.join(save_dir, '%s' % epoch)
        mean_valid = validate(logger, epoch, valid_loader, actor, reward_fn, render_fn,
                              valid_dir, num_plot=5)

        logger.scalar_summary('mean rewards', mean_valid, epoch)
        logger.scalar_summary('loss', mean_loss, epoch)


        # Save best model parameters
        if mean_valid < best_reward:

            best_reward = mean_valid

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

        print('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, took: %2.4fs '\
              '(%2.4fs / 100 batches)\n' % \
              (mean_loss, mean_reward, mean_valid, time.time() - epoch_start,
              np.mean(times)))


def train_gpu(args):

    print('Starting GPU Assignment training')

    GPUS_PER_MACHINE = 4
    MACHINES_PER_RACK = 2
    RACKS_PER_CLUSTER = 2
    # STATIC_SIZE = 23 # (x, y)
    STATIC_SIZE = GPUS_PER_MACHINE * MACHINES_PER_RACK * RACKS_PER_CLUSTER + 1 + \
                  RACKS_PER_CLUSTER + MACHINES_PER_RACK * RACKS_PER_CLUSTER

    STATIC_SIZE = GPUS_PER_MACHINE * MACHINES_PER_RACK * RACKS_PER_CLUSTER + 1

    DYNAMIC_SIZE = 2 # (load, demand)
    NUM_SAMPLES = 1000
    logger = Logger(f'./logs/test')

    train_data = GpuAssignmentDataset(num_samples=NUM_SAMPLES,
                                      gpus_per_machine=GPUS_PER_MACHINE,
                                      machines_per_rack=MACHINES_PER_RACK,
                                      racks_per_cluster=RACKS_PER_CLUSTER)

    print('Train data: {}'.format(train_data))
    valid_data = GpuAssignmentDataset(num_samples=NUM_SAMPLES,
                                      gpus_per_machine=GPUS_PER_MACHINE,
                                      machines_per_rack=MACHINES_PER_RACK,
                                      racks_per_cluster=RACKS_PER_CLUSTER)

    actor = DRL4TSP(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args.hidden_size,
                    train_data.update_dynamic,
                    train_data.update_mask,
                    args.num_layers,
                    args.dropout,
                    train_data.demand_mask).to(device)

    print('Actor: {} '.format(actor))

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)

    print('Critic: {}'.format(critic))

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = train_data.reward
    kwargs['render_fn'] = train_data.render
    kwargs['num_epoch'] = 20

    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, critic, **kwargs)

    test_data = GpuAssignmentDataset(num_samples=NUM_SAMPLES,
                                      gpus_per_machine=GPUS_PER_MACHINE,
                                      machines_per_rack=MACHINES_PER_RACK,
                                      racks_per_cluster=RACKS_PER_CLUSTER)

    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validate(logger, 20, test_loader, actor, train_data.reward, train_data.render, test_dir, num_plot=10)
    print('Average rewards: ', out)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='vrp')
    parser.add_argument('--nodes', dest='num_nodes', default=10, type=int)
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size',default=20, type=int)
    parser.add_argument('--valid-size', default=20, type=int)

    args = parser.parse_args()

    #print('NOTE: SETTTING CHECKPOINT: ')
    #args.checkpoint = os.path.join('vrp', '10', '12_59_47.350165' + os.path.sep)
    #print(args.checkpoint)

    
    
    train_gpu(args)
    