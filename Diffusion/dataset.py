import torch


class Traj(torch.utils.data.Dataset):
    """Some Information about Traj"""
    def __init__(self):
        super(Traj, self).__init__()
        self.datas = torch.cat([torch.ones(100, 17, 10), 
                                torch.ones(100, 17, 256)], dim=-1)

    def __getitem__(self, index):
        return self.datas[index]

    def __len__(self):
        return 100