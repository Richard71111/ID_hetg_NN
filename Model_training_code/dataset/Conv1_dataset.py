from scipy.io import loadmat
from torch.utils.data import Dataset
from torch import tensor
import torch


class Conv1Dataset(Dataset):
    def __init__(self,
                 device,
                 dtype = torch.float32,
                 mode = 'train',
                 Ncell = 50,
                 GJ_coupling = "strong",
                 stats = None
    ):
        super(Conv1Dataset, self).__init__()
        self.device = device
        self.dtype = dtype
        self.Ncell = Ncell
        self.mode  = mode
        self.mesh_num = 384
        self.GJ_coupling = GJ_coupling
        self.stats = stats

        if GJ_coupling.lower() == "strong":
            self.mesh_dir = "/fs/ess/PAS1622/RichardSui/MultiMesh_Cable_Cleft/CNNData/StrongGJ/"
        else:
            self.mesh_dir = "/fs/ess/PAS1622/RichardSui/MultiMesh_Cable_Cleft/CNNData/WeakGJ/"
        self._loadData()
        print(f"Finish loading {mode} data\n")
        
    def _loadData(self):
        train_idx = int(0.8 * self.mesh_num)
        valid_idx = int(0.9 * self.mesh_num)
        input     = []
        target    = []
        if self.mode == 'train':
            mesh_range = range(1, train_idx+1)
        elif self.mode == 'valid':
            mesh_range = range(train_idx+1, valid_idx+1)
        else:
            mesh_range = range(valid_idx+1, self.mesh_num+1)
        for mesh_idx in mesh_range:
            data = self._load_mat(mesh_idx)
            input.append(tensor(data['input'],  dtype=self.dtype))
            target.append(tensor(data['target'], dtype=self.dtype))
        self.inputs = torch.cat(input, dim=0)
        self.labels = torch.cat(target, dim=0)
    def _load_mat(self, mesh_idx):
        if self.GJ_coupling.lower() == "strong":
            mesh_path = self.mesh_dir + f"FEMDATA_{mesh_idx}mat_StrongGJ.mat"
        else:
            mesh_path = self.mesh_dir + f"FEMDATA_{mesh_idx}mat_WeakGJ.mat"
        data = loadmat(mesh_path)
        return data
    def normalize(self):
        if self.mode == 'train':
            Inmean = self.inputs.mean(dim=0, keepdim=True)
            Instd  = self.inputs.std(dim=0, keepdim=True) + 1e-8

            Outmean = self.labels.mean(dim=0, keepdim=True)
            Outstd  = self.labels.std(dim=0, keepdim=True) + 1e-8

            self.inputs = (self.inputs - Inmean) / Instd
            self.labels = (self.labels - Outmean) / Outstd

            self.stats = {'Inmean': Inmean.cpu(), 'Instd': Instd.cpu(), 'Outmean': Outmean.cpu(), 'Outstd': Outstd.cpu()}
        
        else:
            assert self.stats is not None, "Need stats (mean/std) for normalization in valid/test!"
            Inmean = self.stats['Inmean'].to(self.dtype)
            Instd  = self.stats['Instd'].to(self.dtype)
            Outmean = self.stats['Outmean'].to(self.dtype)
            Outstd  = self.stats['Outstd'].to(self.dtype)
            self.inputs = (self.inputs - Inmean) / Instd
            self.labels = (self.labels - Outmean) / Outstd
    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        x = self.inputs[idx, :, 1:-1]
        y = self.labels[idx, :, 1:-1]
        if self.mode == 'train':
            noise_std = 0.01
            noise = torch.randn_like(x) * noise_std
            x = x + noise
        return x,y

class TestDataset(Dataset):
    def __init__(self,
                 device,
                 dtype = torch.float32,
                 mode = 'train',
                 Ncell = 50,
                 GJ_coupling = "strong",
                 stats = None
    ):
        super(TestDataset, self).__init__()
        self.X = torch.randn(5000,7,50,dtype=dtype)
        self.Y = torch.randn(5000,2,49,dtype=dtype)
        self.stats = torch.rand(1,1,1)
        self.device = device
        self.dtype = dtype

    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx],self.Y[idx]
    def normalize(self):
        pass