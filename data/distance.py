# Code from: https://ai.stackexchange.com/questions/36191/how-to-calculate-a-meaningful-distance-between-multidimensional-tensors
import


def cumsum_3d(a):
    a = torch.cumsum(a, -1)
    a = torch.cumsum(a, -2)
    a = torch.cumsum(a, -3)
    return a

def norm_3d(a):
    return a / torch.sum(a, dim=(-1,-2,-3), keepdim=True)

def emd_3d(a, b):
    a = norm_3d(a)
    b = norm_3d(b)
    return torch.mean(torch.square(cumsum_3d(a) - cumsum_3d(b)), dim=(-1,-2,-3))