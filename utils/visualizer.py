import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.distributions import multivariate_normal
from model.loss import generate_statistics_matrices


def figure_to_array(fig):
    r"""Convert plt.figure to RGBA numpy array. shape: height, width, layer"""

    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)


def data_visualizer(V_pred, obs_traj, pred_traj_gt, samples=1000, n_levels=30):
    # generate gt trajectory
    V_gt = torch.cat((obs_traj, pred_traj_gt), dim=3).squeeze(dim=0).permute(2, 0, 1)

    # trajectory sampling
    mu, cov = generate_statistics_matrices(V_pred.squeeze(dim=0))
    mv_normal = multivariate_normal.MultivariateNormal(mu, cov)
    V_smpl = mv_normal.sample((samples,))

    # relative points to absolute points
    V_absl = []
    for t in range(V_smpl.size(1)):
        V_absl.append(V_smpl[:, 0:t+1, :, :].sum(dim=1, keepdim=True) + V_gt[7, :, :])
    V_absl = torch.cat(V_absl, dim=1)

    # visualize trajectories
    V_absl_temp = V_absl.view(-1, V_absl.size(2), 2)[:, :, :].cpu().numpy()
    V_gt_temp = V_gt[:, :, :].cpu().numpy()

    fig = plt.figure(figsize=(10, 7))

    for n in range(V_smpl.size(2)):
        ax = sns.kdeplot(V_absl_temp[:, n, 0], V_absl_temp[:, n, 1], n_levels=n_levels, shade=True, shade_lowest=False)
        plt.plot(V_gt_temp[:, n, 0], V_gt_temp[:, n, 1], linestyle='--', color='C{}'.format(n), linewidth=1)

    ax.tick_params(axis="y", direction="in", pad=-22)
    ax.tick_params(axis="x", direction="in", pad=-15)
    plt.xlim(-14, 36)
    plt.ylim(-9, 26)
    plt.tight_layout()

    plt.close()

    return figure_to_array(fig)


def visualize_scene(data_loader, model, frame_id):
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx == frame_id:
            V_obs, A_obs, V_tr, A_tr = [tensor.cuda() for tensor in batch[-4:]]
            obs_traj, pred_traj_gt = [tensor.cuda() for tensor in batch[:2]]

            V_obs_ = V_obs.permute(0, 3, 1, 2)
            V_pred, _ = model(V_obs_, A_obs)
            V_pred = V_pred.permute(0, 2, 3, 1)

            V_pred = V_pred.squeeze()
            V_obs_traj = obs_traj.permute(0, 3, 1, 2).squeeze(dim=0)
            V_pred_traj_gt = pred_traj_gt.permute(0, 3, 1, 2).squeeze(dim=0)

            # Visualize trajectories
            fig_img = data_visualizer(V_pred.unsqueeze(dim=0), obs_traj, pred_traj_gt)

            plt.imshow(fig_img[:, :, :3])
            plt.axis('off')
            plt.tight_layout()
            plt.show()

            break
