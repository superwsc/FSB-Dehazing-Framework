import torch

class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor   
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor


class MixUp_AUG:
    def __init__(self):
        #beta分布
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        #生成一个长度为 bs 的随机排列的索引，这将用于对样本进行随机排列。
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]
#从预定义的分布 self.dist 中抽样生成一个随机数 lam，这个随机数在 [0, 1] 之间，表示数据增强的强度。rsample 是 PyTorch 分布对象的一个方法，用于生成样本。
#self.dist.rsample((bs, 1)): 从 Beta 分布中抽取样本，参数 (bs, 1) 指定了要生成的样本数量和样本的形状。这里 bs 是 batch size，而 (1) 表示每个样本都是一个标量值。
#.view(-1, 1, 1, 1): 这一步对生成的样本进行形状变换。-1 表示自动推断该维度的大小，而 (1, 1, 1) 则指定了新的形状，这里是为了将样本形状变为 (bs, 1, 1, 1)，以便与数据进行混合。
        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2

        return rgb_gt, rgb_noisy
