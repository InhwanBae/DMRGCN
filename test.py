import pickle
import argparse
import torch

from tqdm import tqdm
from utils import TrajectoryDataset
from model import *

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import multivariate_normal

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='tag', help='Personal tag for the model')
parser.add_argument('--n_samples', type=int, default=20, help='Number of samples')
parser.add_argument('--visualize', action="store_true", default=False, help='Visualize trajectories')
test_args = parser.parse_args()

# Get arguments for training
checkpoint_dir = './checkpoints/' + test_args.tag + '/'

args_path = checkpoint_dir + '/args.pkl'
with open(args_path, 'rb') as f:
    args = pickle.load(f)

dataset_path = './datasets/' + args.dataset + '/'
model_path = checkpoint_dir + args.dataset + '_best.pth'
KSTEPS = test_args.n_samples

# Data preparation
test_dataset = TrajectoryDataset(dataset_path + 'test/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# Model preparation
model = social_dmrgcn(n_stgcn=args.n_stgcn, n_tpcnn=args.n_tpcnn,
                      output_feat=args.output_size, kernel_size=args.kernel_size,
                      seq_len=args.obs_seq_len, pred_seq_len=args.pred_seq_len)
model = model.cuda()
model.load_state_dict(torch.load(model_path))

# Test logging
writer = SummaryWriter(checkpoint_dir)
if test_args.visualize:
    from utils import data_visualizer


def test(KSTEPS=20):
    model.eval()

    ade_all = []
    fde_all = []

    progressbar = tqdm(range(len(test_loader)))
    progressbar.set_description('Testing {}'.format(test_args.tag))

    for batch_idx, batch in enumerate(test_loader):
        V_obs, A_obs, V_tr, A_tr = [tensor.cuda() for tensor in batch[-4:]]
        obs_traj, pred_traj_gt = [tensor.cuda() for tensor in batch[:2]]

        V_obs_ = V_obs.permute(0, 3, 1, 2)
        V_pred, _ = model(V_obs_, A_obs)
        V_pred = V_pred.permute(0, 2, 3, 1)

        V_pred = V_pred.squeeze()
        V_obs_traj = obs_traj.permute(0, 3, 1, 2).squeeze(dim=0)
        V_pred_traj_gt = pred_traj_gt.permute(0, 3, 1, 2).squeeze(dim=0)

        # Randomly sampling predict trajectories
        mu, cov = generate_statistics_matrices(V_pred.squeeze(dim=0))
        mv_normal = multivariate_normal.MultivariateNormal(mu, cov)
        V_pred_sample = mv_normal.sample((KSTEPS,))

        # Relative trajectories to absolute trajectories
        V_absl = []
        for t in range(V_pred_sample.size(1)):
            V_absl.append(V_pred_sample[:, 0:t + 1, :, :].sum(dim=1, keepdim=True) + V_obs_traj[-1, :, :])
        V_absl = torch.cat(V_absl, dim=1)

        # Calculate ADEs and FDEs for each trajectory
        temp = V_absl - V_pred_traj_gt
        temp = (temp ** 2).sum(dim=-1).sqrt()

        ADEs = temp.mean(dim=1).min(dim=0)[0]
        FDEs = temp[:, -1, :].min(dim=0)[0]

        ade_all.extend(ADEs.tolist())
        fde_all.extend(FDEs.tolist())

        # Visualize trajectories
        if test_args.visualize and batch_idx % 1 == 0:
            fig_img = data_visualizer(V_pred.unsqueeze(dim=0), obs_traj, pred_traj_gt, samples=100)
            writer.add_image('Test', fig_img[:, :, :], batch_idx, dataformats='HWC')

        progressbar.update(1)

    progressbar.close()

    ade_ = sum(ade_all) / len(ade_all)
    fde_ = sum(fde_all) / len(fde_all)

    return ade_, fde_


def main():
    ade, fde = test(KSTEPS)

    result_lines = ["Evaluating model: {}".format(test_args.tag),
                    "ADE: {0}, FDE: {1}".format(ade, fde)]

    with open(checkpoint_dir + 'results.txt', 'a') as f:
        for line in result_lines:
            f.write(line + '\n')
            print(line)


if __name__ == "__main__":
    main()

writer.close()
