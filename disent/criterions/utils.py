import torch


def get_gradient_penalty(x, y, model):
    alpha = torch.rand((x.size(0), 1), device=x.device)
    interpolates = alpha * x + (1 - alpha) * y
    f_int = model(interpolates).sum()
    grads = torch.autograd.grad(f_int, interpolates, create_graph=model.training)[0]
    slopes = grads.pow(2).sum(1).sqrt()
    gp = torch.mean((slopes - 1) ** 2)
    return gp


def mmd_rbf(posterior_sample, prior_sample, prior_variance):
    """rbf kernel for samples from two distributions"""
    # x, y: (bsz, csz); var: float
    x, y, var = posterior_sample, prior_sample, prior_variance
    bsz, csz = x.size()
    # size: (csz, csz)
    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    C = 2. * csz * prior_variance

    # rbf kernels: \sum_{ij}(\exp{-|x_i-x_j|^2/C})
    kxx = torch.sum(torch.exp(- (rx.t() + rx - 2 * xx) / C))
    kyy = torch.sum(torch.exp(- (ry.t() + ry - 2 * yy) / C))
    kxy = torch.sum(torch.exp(- (rx.t() + ry - 2 * xy) / C))
    
    mmd = (kxx + kyy) / (bsz * (bsz - 1)) + 2. * kxy / (bsz * bsz)
    return mmd


def mmd_imq(posterior_sample, prior_sample, prior_variance):
    x, y, var = posterior_sample, prior_sample, prior_variance
    bsz, csz = x.size()
    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    C = 2. * csz * prior_variance

    # imq kernels: \sum_{ij}[C/(C+|x_i-x_j|^2)]
    kxx = torch.sum(C / (rx.t() + rx - 2 * xx + C))
    kyy = torch.sum(C / (ry.t() + ry - 2 * yy + C))
    kxy = torch.sum(C / (rx.t() + ry - 2 * xy + C))

    mmd = (kxx + kyy) / (bsz * (bsz - 1)) + 2. * kxy / (bsz * bsz)
    return mmd


def shuffle_code(code):
    """Shuffle latent variables across the batch

    Args:
        code: [batch_size, code_size]
    """
    code = code.clone()
    shuffled = []
    bsz, csz = code.size()
    for i in range(csz):
        idx = torch.randperm(bsz)
        shuffled.append(code[idx][:, i])
    return torch.stack(shuffled, dim=1)
