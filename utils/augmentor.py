import math
import random
import torch


def data_sampler(V_obs, A_obs, V_tr, A_tr, batch=4, scale=True, stretch=True, flip=True, rotation=True, noise=True):
    r"""Returns the Trajectories with batch size."""

    aug_Vo, aug_Ao, aug_Vg, aug_Ag = [], [], [], []

    for i in range(batch):
        V_obs_t, A_obs_t, V_tr_t, A_tr_t = V_obs.clone(), A_obs.clone(), V_tr.clone(), A_tr.clone()
        if scale:
            V_obs_t, A_obs_t, V_tr_t, A_tr_t = random_scale(V_obs_t, A_obs_t, V_tr_t, A_tr_t, min=0.8, max=1.2)
        if stretch:
            V_obs_t, A_obs_t, V_tr_t, A_tr_t = random_stretch(V_obs_t, A_obs_t, V_tr_t, A_tr_t, min=0.5, max=2.0)
        if flip:
            V_obs_t, A_obs_t, V_tr_t, A_tr_t = random_flip(V_obs_t, A_obs_t, V_tr_t, A_tr_t)
        if rotation:
            V_obs_t, A_obs_t, V_tr_t, A_tr_t = random_rotation(V_obs_t, A_obs_t, V_tr_t, A_tr_t)
        if noise:
            V_obs_t, A_obs_t, V_tr_t, A_tr_t = random_noise(V_obs_t, A_obs_t, V_tr_t, A_tr_t)

        aug_Vo.append(V_obs_t.squeeze(dim=0))
        aug_Ao.append(A_obs_t.squeeze(dim=0))
        aug_Vg.append(V_tr_t.squeeze(dim=0))
        aug_Ag.append(A_tr_t.squeeze(dim=0))

    V_obs = torch.stack(aug_Vo).detach()
    A_obs = torch.stack(aug_Ao).detach()
    V_tr = torch.stack(aug_Vg).detach()
    A_tr = torch.stack(aug_Ag).detach()

    return V_obs, A_obs, V_tr, A_tr


def random_scale(V_obs, A_obs, V_tr, A_tr, min=0.9, max=1.1):
    r"""Returns the randomly stretched Trajectories."""

    scale = [random.uniform(min, max), random.uniform(min, max)]
    scale = torch.tensor(scale).cuda()
    scale_a = torch.sqrt(scale[0] * scale[1])
    return V_obs * scale, A_obs * scale_a, V_tr * scale, A_tr * scale_a


def random_stretch(V_obs, A_obs, V_tr, A_tr, min=0.9, max=1.1):
    r"""Returns the randomly stretched Trajectories."""

    scale = [random.uniform(min, max), random.uniform(min, max)]
    scale = torch.tensor(scale).cuda()
    scale_a = torch.sqrt(scale[0] * scale[1])
    return V_obs * scale, A_obs * scale_a, V_tr * scale, A_tr * scale_a


def random_flip(V_obs, A_obs, V_tr, A_tr):
    r"""Returns the randomly flipped Trajectories."""

    flip = random.choice([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    flip = torch.tensor(flip).cuda()

    for i in range(1, (A_obs.size(1) // 2)):
        A_obs[:, i*2] *= flip[0]
        A_tr[:, i*2] *= flip[0]
        A_obs[:, i*2+1] *= flip[1]
        A_tr[:, i*2+1] *= flip[1]

    return V_obs * flip, A_obs, V_tr * flip, A_tr


def random_rotation(V_obs, A_obs, V_tr, A_tr):
    r"""Returns the randomly rotated Trajectories."""

    theta = random.uniform(-math.pi, math.pi)
    theta = (theta // (math.pi/2)) * (math.pi/2)

    r_mat = [[math.cos(theta), -math.sin(theta)],
             [math.sin(theta), math.cos(theta)]]
    r = torch.tensor(r_mat, dtype=torch.float, requires_grad=False).cuda()

    V_obs = torch.einsum('rc,ntvc->ntvr', r, V_obs)
    V_tr = torch.einsum('rc,ntvc->ntvr', r, V_tr)

    for i in range(1, (A_obs.size(1) // 2)):
        A_obs[:, i*2:i*2+2] = torch.einsum('rc,nctvw->nrtvw', r, A_obs[:, i*2:i*2+2])
        A_tr[:, i*2:i*2+2] = torch.einsum('rc,nctvw->nrtvw', r, A_tr[:, i*2:i*2+2])

    return V_obs, A_obs, V_tr, A_tr


def random_noise(V_obs, A_obs, V_tr, A_tr, std=0.01):
    r"""Returns the randomly noised Trajectories."""

    noise_obs = torch.randn_like(V_obs) * std
    noise_tr = torch.randn_like(V_tr) * std
    return V_obs + noise_obs, A_obs, V_tr + noise_tr, A_tr
