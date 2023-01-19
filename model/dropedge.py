import torch


def drop_edge(A, percent, training=True, inplace=False):
    r"""Returns the randomly dropped edge Adjacency matrix with preserve rate."""

    assert 0 <= percent <= 1.0
    if not training:
        return A
    A_prime = torch.rand_like(A)
    A_drop = A if inplace else A.clone()
    A_drop[A_prime > percent] = 0
    return A_drop
