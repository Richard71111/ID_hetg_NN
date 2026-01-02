import torch
import math
import time
import joblib
from PermanentConstants import NormalizationCNN

class CouplingBase:
    def compute(self, phi_i, ti=None):
        raise NotImplementedError("CouplingBase is an abstract class.")

class ResistorCoupling(CouplingBase):
    def __init__(self, N, Ggap, device, dtype):
        self.N = N
        self.Ggap = Ggap
        self.device = device
        self.dtype = dtype
        self.I = torch.zeros(N, device=device, dtype=dtype)
        self.bc = round(0.25 * N)  

    def compute(self, phi_i, ti=None):
        I = self.I
        G = self.Ggap

        # 1D nearest-neighbor diffusion coupling
        I[1:-1] = G * (2*phi_i[1:-1] - phi_i[:-2] - phi_i[2:])
        I[0]    = G * (phi_i[0]  - phi_i[1])
        I[-1]   = G * (phi_i[-1] - phi_i[-2])
        return I

class ResNetSimpleCoupling(CouplingBase):
    def __init__(self, 
                 N, 
                 Ggap, 
                 boundary, 
                 model,
                 GJ_coupling='SingleGJ',
                 model_name = "ResNet", 
                 if_normalize=None, 
                 device=None, 
                 dtype=torch.float32):
        self.N = N
        self.Ggap = Ggap
        self.bc = int(boundary)
        self.slices = slice(self.bc, -self.bc)
        self.model = model.to(device=device, dtype=dtype)
        self.GJ_coupling = GJ_coupling
        self.model_path = f"Model_state/{model_name}/best_{model_name}_{GJ_coupling}_fullseq.pth"
        self.if_normalize = if_normalize
        self.device = device
        self.dtype = dtype
        self._load_model()
        if self.if_normalize is not None:
            self._load_scaler()
            self.normalize = NormalizationCNN(self.scaler, slices=self.slices, device=device, dtype=dtype)

        self.I = torch.zeros(N, device=device, dtype=dtype)
    def _load_model(self):
        best_state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(best_state)
        self.model.eval()
    def _load_scaler(self):
        if self.GJ_coupling == 'strong':
            GJ = 735  # nS
            scaler_name = 'Scaler/CNN/strong_standard_scaler.pkl'
        elif self.GJ_coupling == 'weak':
            GJ = 73.5 # nS
            scaler_name = 'Scaler/CNN/weak_standard_scaler.pkl'
        else:
            GJ = 7.35 # nS
        scaler_name = 'Scaler/CNN/SingleGJ_standard_scaler.pkl'
        self.scaler = joblib.load(scaler_name)

    def compute(self, phi_i, ti=None):
        bc = self.bc
        Ggap = self.Ggap
        I_gap = self.I

        # ---- NN cleft current ----
        phi_pad = phi_i[self.slices].reshape(1, 1, -1)  # (1,1,L)
        if self.if_normalize is not None:
            phi_pad = self.normalize.NormalizeInput(phi_pad)
        I_cleft = self.model(phi_pad)
        if self.if_normalize is not None:
            I_cleft = self.normalize.DenormalizeOutput(I_cleft)
        I_cleft = I_cleft.squeeze(0)  #  (1,L) or (2,L)

        # ---- bulk Laplacian outside cleft region ----
        I_gap[1:bc] = Ggap * (phi_i[1:bc] - phi_i[:bc-1] + phi_i[1:bc] - phi_i[2:bc+1])
        I_gap[-bc:-1] = Ggap * (phi_i[-bc:-1] - phi_i[-bc-1:-2] + phi_i[-bc:-1] - phi_i[-bc+1:])

        # boundaries
        I_gap[0]  = Ggap * (phi_i[0]  - phi_i[1])
        I_gap[-1] = Ggap * (phi_i[-1] - phi_i[-2])

        # interfaces to cleft region
        I_gap[bc]    = Ggap * (phi_i[bc]    - phi_i[bc-1])
        I_gap[-bc-1] = Ggap * (phi_i[-bc-1] - phi_i[-bc])

        # ---- add cleft currents ----
        I_gap[bc:-bc-1]   += -I_cleft[0, :]
        I_gap[bc+1:-bc]   +=  I_cleft[0, :]

        return I_gap

class MLPSimpleCoupling(CouplingBase):
    def __init__(self, 
                 N, 
                 Ggap, 
                 boundary, 
                 model,
                 GJ_coupling='SingleGJ',
                 model_name = "MLP",
                 scaler=None,  
                 device=None, 
                 dtype=torch.float32):
        self.N = N
        self.Ggap = Ggap
        self.bc = int(boundary)
        self.slices = slice(self.bc, -self.bc)
        self.model = model.to(device=device, dtype=dtype)
        self.GJ_coupling = GJ_coupling
        self.count = 1
        # self.model_path = f"Model_state/{model_name}/best_{model_name}_{GJ_coupling}_fullseq.pth"
        self.model_path = f"Model_state/{model_name}/best_{model_name}_SingleGJ_adaptive_norm.pth"
        self.scaler = scaler
        if scaler is not None:
            self.scaler_x = scaler['scaler_X']
            self.scaler_y = scaler['scaler_Y']
        self.device = device
        self.dtype = dtype
        self._load_model()

        self.I = torch.zeros(N, device=device, dtype=dtype)
    def _load_model(self):
        best_state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(best_state)
        self.model.eval()

    def _reshape(self, phi):
        assert phi.ndim == 2, "phi must be 2D (T, Ncell)"
        _, Ncell = phi.shape
        assert Ncell >= 2, "Need at least 2 cells"

        v_left  = phi[:, :-1]    # (T, Ncell-1)
        v_right = phi[:, 1:]     # (T, Ncell-1)

        pairs = torch.stack([v_left, v_right], dim=2)  # (T, Ncell-1, 2)

        return pairs.reshape(-1, 2)

    def compute(self, phi_i, ti=None):
        
        Ggap = self.Ggap
        I_gap = self.I
        bc = self.bc

        # ---------- 1. NN edge current J ----------
        # phi_NN shape: (M-1, 2)  for edges (bc+k, bc+k+1)
        phi_NN = self._reshape(phi_i[self.slices].unsqueeze(0))

        if self.scaler is not None:
            phi_NN = self.scaler_x.transform(phi_NN.cpu().numpy())
            phi_NN = torch.from_numpy(phi_NN).to(self.device, self.dtype)

        J = self.model(phi_NN)    # edge currents

        if self.scaler is not None:
            J = self.scaler_y.inverse_transform(J.cpu().numpy())
            J = torch.from_numpy(J).to(self.device, self.dtype)

        J = J.view(-1)            # shape (M-1,)


        # ---------- 2. bulk Laplacian outside NN region ----------
        I_gap.zero_()

        I_gap[1:bc] = Ggap * (phi_i[1:bc] - phi_i[:bc-1] + phi_i[1:bc] - phi_i[2:bc+1])
        I_gap[-bc:-1] = Ggap * (phi_i[-bc:-1] - phi_i[-bc-1:-2] + phi_i[-bc:-1] - phi_i[-bc+1:])

        # boundaries
        I_gap[0]  = Ggap * (phi_i[0]  - phi_i[1])
        I_gap[-1] = Ggap * (phi_i[-1] - phi_i[-2])

        # interfaces to NN region (left and right)
        I_gap[bc]    = Ggap * (phi_i[bc]    - phi_i[bc-1])
        I_gap[-bc-1] = Ggap * (phi_i[-bc-1] - phi_i[-bc])


        # ---------- 3. Add NN flux via discrete divergence ----------
        # NN nodes: i = bc ... bc+M-1
        # J[k] is flux from (bc+k) -> (bc+k+1)

        M = phi_i[self.slices].numel()

        # leftmost NN node
        I_gap[bc] += -J[0]

        # interior NN nodes
        for k in range(1, M-1):
            i = bc + k
            I_gap[i] += J[k-1] - J[k]

        # rightmost NN node
        I_gap[bc + M - 1] += J[M-2]

        return I_gap

class CableSimulator:
    def __init__(self, const, coupling, Ord11_model_fn, device, dtype, Vthresh = -60, dt=0.01, save_stride=100):
        self.const = const
        self.parameters = const.parameters
        self.coupling = coupling
        self.Ord11_model_fn = Ord11_model_fn
        self.Vthresh = Vthresh
        self.device = device
        self.dtype = dtype

        self.dt = float(dt)
        self.save_stride = int(save_stride)

        self.N = self.parameters["N"]
        self.Ctot = self.parameters["Ctot"]
        self.reset()

    def reset(self):
        self.ti = 0.0
        self.count = 0
        self.phi_i = self.const.phi0.clone()
        self.G_i = self.const.g0.clone()
        self.S = self.const.S.clone()
        self.state = torch.cat((self.phi_i, self.G_i), dim=0)
        self.phi_save, self.I_save = [], []
        self.last_print_time = 0.0
        self.beat_num = [0 for _ in range(self.N)]   # 0-based
        self.tup      = [[] for _ in range(self.N)]
        self.trepol   = [[] for _ in range(self.N)]

    def update_tup_repol(self,):
        Vm = self.phi_i
        Vm_old = self.phi_old

        for i in range(self.N):

            # ---------- upstroke ----------
            if (Vm[i] > self.Vthresh) and (Vm_old[i] < self.Vthresh):
                y1 = Vm_old[i]
                y2 = Vm[i]
                m  = (y2 - y1) / self.dt
                t_cross = self.ti - (y1 - self.Vthresh) / m

                b = self.beat_num[i]
                if b == len(self.tup[i]):
                    self.tup[i].append(t_cross)
                else:
                    self.tup[i][b] = t_cross

            # ---------- downstroke ----------
            if (Vm[i] < self.Vthresh) and (Vm_old[i] > self.Vthresh):
                b = self.beat_num[i]
                if b < len(self.tup[i]) and (self.ti - self.tup[i][b]) > 0.1:
                    y1 = Vm_old[i]
                    y2 = Vm[i]
                    m  = (y2 - y1) / self.dt
                    t_cross = self.ti - (y1 - self.Vthresh) / m

                    if b == len(self.trepol[i]):
                        self.trepol[i].append(t_cross)
                    else:
                        self.trepol[i][b] = t_cross

                    self.beat_num[i] += 1
    def compute_CV(self):
        i1 = self.coupling.bc - 2
        i2 = self.N - 1 - i1
        dx = self.parameters["L"] / 1000.0  # mm

        t1 = self.tup[i1][0]
        t2 = self.tup[i2][0]

        return dx * (i2 - i1) / (t2 - t1)
    
    def step(self, dt=None):
        if dt is not None:
            self.dt = float(dt)
        self.parameters["dt"] = self.dt

        I_couple = self.coupling.compute(self.phi_i, self.ti)
        self.phi_old = self.phi_i.clone()
        Iion, _, G_new, _ = self.Ord11_model_fn(
            self.ti, self.state, self.parameters, self.S, self.device, self.dtype
        )
        if torch.isnan(Iion).any():
            raise FloatingPointError(f"NaN in Iion at time {self.ti}")
                
        self.phi_i = self.state[0:self.N] - self.dt / self.Ctot * (Iion + I_couple)
        if torch.isnan(self.phi_i).any():
            raise FloatingPointError(f"NaN in phi at time {self.ti}")

        self.state[0:self.N] = self.phi_i
        self.state[self.N:] = G_new
        self.update_tup_repol()

        self.ti = round(self.ti + self.dt, 5) # avoid floating point error accumulation
        self.count += 1
        return I_couple
    def adaptive_step(self, ti):
        bcl = float(self.parameters["bcl"])
        if abs(math.fmod(ti, bcl)) < self.const.twin:
            return float(self.const.dt1), float(self.const.dt1_samp)
        else:
            return float(self.const.dt2), float(self.const.dt2_samp)
    def run(self, T=None, print_every=1000.0):
        if T is None:
            T = float(self.const.T)

        self.phi_save, self.I_save, self.t_save = [], [], []
        self.next_sample_time = 0.0

        start = time.time()
        with torch.no_grad():
            while self.ti < T:
                dt, dt_samp = self.adaptive_step(self.ti)
                if self.ti + dt > T:
                    dt = T - self.ti
                    if dt <= 0:
                        break

                if print_every is not None and (self.ti - self.last_print_time >= float(print_every)):
                    print(f"time: {self.ti:.2f} ms, dt={dt:.4g}")
                    self.last_print_time += float(print_every)

                I_couple = self.step(dt)
                if self.ti + 1e-12 >= self.next_sample_time:
                    self.phi_save.append(self.phi_i.clone())
                    self.I_save.append(I_couple.clone())
                    self.t_save.append(self.ti)
                    self.next_sample_time += dt_samp
        self.cv = self.compute_CV()
        phi_save = torch.stack(self.phi_save, dim=0) if self.phi_save else torch.empty(0)
        I_save   = torch.stack(self.I_save, dim=0) if self.I_save else torch.empty(0)
        t_save   = torch.tensor(self.t_save, device=self.device, dtype=self.dtype) if self.t_save else torch.empty(0)

        print(f"Simulation time: {time.time() - start:.2f} seconds")
        return t_save, phi_save, I_save, self.cv

