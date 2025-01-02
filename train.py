import os
import pickle
import argparse
import torch

from tqdm import tqdm
from model import *
from utils import TrajectoryDataset, data_sampler

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# To avoid contiguous problem.
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# Argument parsing
parser = argparse.ArgumentParser()

# Model specific parameters
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_stgcn', type=int, default=1, help='Number of GCN layers')
parser.add_argument('--n_tpcnn', type=int, default=4, help='Number of CNN layers')
parser.add_argument('--kernel_size', type=int, default=3)

# Data specific parameters
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)
parser.add_argument('--dataset', default='eth', help='Dataset name(eth,hotel,univ,zara1,zara2)')

# Training specific parameters
parser.add_argument('--batch_size', type=int, default=128, help='Mini batch size')
parser.add_argument('--num_epochs', type=int, default=128, help='Number of epochs')
parser.add_argument('--clip_grad', type=float, default=None, help='Gradient clipping')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=32, help='Number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true", default=False, help='Use lr rate scheduler')
parser.add_argument('--tag', default='tag', help='Personal tag for the model')
parser.add_argument('--visualize', action="store_true", default=False, help='Visualize trajectories')

args = parser.parse_args()

# Data preparation
# Batch size set to 1 because vertices vary by humans in each scene sequence.
# Use mini batch working like batch.
dataset_path = './datasets/' + args.dataset + '/'
checkpoint_dir = './checkpoints/' + args.tag + '/'

train_dataset = TrajectoryDataset(dataset_path + 'train/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

val_dataset = TrajectoryDataset(dataset_path + 'val/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# Model preparation
model = social_dmrgcn(n_stgcn=args.n_stgcn, n_tpcnn=args.n_tpcnn,
                      output_feat=args.output_size, kernel_size=args.kernel_size,
                      seq_len=args.obs_seq_len, pred_seq_len=args.pred_seq_len)
model = model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
if args.use_lrschd:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.8)

# Train logging
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
with open(checkpoint_dir + 'args.pkl', 'wb') as f:
    pickle.dump(args, f)

writer = SummaryWriter(checkpoint_dir)
if args.visualize:
    from utils import data_visualizer
metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 1e10}


def train(epoch):
    global metrics
    model.train()
    loss_batch = 0.
    loader_len = len(train_loader)

    progressbar = tqdm(range(loader_len))
    progressbar.set_description('Train Epoch: {0} Loss: {1:.8f}'.format(epoch, 0))

    for batch_idx, batch in enumerate(train_loader):
        # Sum gradients till idx reach to batch_size
        if batch_idx % args.batch_size == 0:
            optimizer.zero_grad()

        V_obs, A_obs, V_tr, A_tr = [tensor.cuda() for tensor in batch[-4:]]

        # Try augmentation to generate a batch.
        aug = True
        if aug:
            V_obs, A_obs, V_tr, A_tr = data_sampler(V_obs, A_obs, V_tr, A_tr, batch=4)

        V_obs_ = V_obs.permute(0, 3, 1, 2)
        V_pred, _ = model(V_obs_, A_obs)
        V_pred = V_pred.permute(0, 2, 3, 1)

        loss = multivariate_loss(V_pred, V_tr, training=True)
        loss.backward()
        loss_batch += loss.item()

        if batch_idx % args.batch_size + 1 == args.batch_size or batch_idx + 1 == loader_len:
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            iter_idx = epoch * loader_len + batch_idx
            writer.add_scalar('Loss/Train_V', (loss_batch / batch_idx), iter_idx)

        progressbar.set_description('Train Epoch: {0} Loss: {1:.8f}'.format(epoch, loss.item() / args.batch_size))
        progressbar.update(1)

    progressbar.close()

    metrics['train_loss'].append(loss_batch / loader_len)


def valid(epoch):
    global metrics, constant_metrics
    model.eval()
    loss_batch = 0.
    loader_len = len(val_loader)

    progressbar = tqdm(range(loader_len))
    progressbar.set_description('Valid Epoch: {0} Loss: {1:.8f}'.format(epoch, 0))

    for batch_idx, batch in enumerate(val_loader):
        # sum gradients till idx reach to batch_size
        if batch_idx % args.batch_size == 0:
            optimizer.zero_grad()

        V_obs, A_obs, V_tr, A_tr = [tensor.cuda() for tensor in batch[-4:]]
        obs_traj, pred_traj_gt = [tensor.cuda() for tensor in batch[:2]]

        V_obs_ = V_obs.permute(0, 3, 1, 2)
        V_pred, _ = model(V_obs_, A_obs)
        V_pred = V_pred.permute(0, 2, 3, 1)

        loss = multivariate_loss(V_pred, V_tr)
        loss_batch += loss.item()

        if batch_idx % args.batch_size + 1 == args.batch_size or batch_idx + 1 == loader_len:
            # Visualize trajectories
            if args.visualize:
                fig_img = data_visualizer(V_pred, obs_traj, pred_traj_gt, samples=100)
                writer.add_image('Valid_{0:04d}'.format(batch_idx), fig_img[:, :, :], epoch, dataformats='HWC')

            iter_idx = epoch * loader_len + batch_idx
            writer.add_scalar('Loss/Valid_V', (loss_batch / batch_idx), iter_idx)

        progressbar.set_description('Valid Epoch: {0} Loss: {1:.8f}'.format(epoch, loss.item() / args.batch_size))
        progressbar.update(1)

    progressbar.close()

    metrics['val_loss'].append(loss_batch / loader_len)

    # Save model
    torch.save(model.state_dict(), checkpoint_dir + args.dataset + '.pth')
    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + args.dataset + '_best.pth')


def main():
    for epoch in range(args.num_epochs):
        train(epoch)
        valid(epoch)
        if args.use_lrschd:
            scheduler.step()

        print(" ")
        print("Dataset: {0}, Epoch: {1}".format(args.tag, epoch))
        print("Train_loss: {0}, Val_los: {1}".format(metrics['train_loss'][-1], metrics['val_loss'][-1]))
        print("Min_val_epoch: {0}, Min_val_loss: {1}".format(constant_metrics['min_val_epoch'],
                                                             constant_metrics['min_val_loss']))
        print(" ")

        with open(checkpoint_dir + 'metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f)

        with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as f:
            pickle.dump(constant_metrics, f)


if __name__ == "__main__":
    main()

writer.close()
