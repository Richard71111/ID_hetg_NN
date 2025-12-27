import torch
import math
import time

class ResistorCoupling:
    def __init__(self, N, Ggap, device, dtype):
        self.N = N
        self.Ggap = Ggap
        self.device = device
        self.dtype = dtype
        self.I = torch.zeros(N, device=device, dtype=dtype)

    def compute(self, phi_i, ti=None):
        I = self.I
        G = self.Ggap

        # 1D nearest-neighbor diffusion coupling
        I[1:-1] = G * (2*phi_i[1:-1] - phi_i[:-2] - phi_i[2:])
        I[0]    = G * (phi_i[0]  - phi_i[1])
        I[-1]   = G * (phi_i[-1] - phi_i[-2])
        return I
    
class NNCleftCoupling:
    def __init__(self, N, Ggap, boundary, model, normalize=None, device=None, dtype=torch.float32):
        self.N = N
        self.Ggap = Ggap
        self.bc = int(boundary)
        self.slices = slice(self.bc, -self.bc)

        self.model = model
        self.normalize = normalize
        self.device = device
        self.dtype = dtype

        self.I = torch.zeros(N, device=device, dtype=dtype)

    def compute(self, phi_i, ti=None):
        bc = self.bc
        Ggap = self.Ggap
        I_gap = self.I

        # ---- NN cleft current ----
        phi_pad = phi_i[self.slices].reshape(1, 1, -1)  # (1,1,L)
        if self.normalize is not None:
            phi_pad = self.normalize.NormalizeInput(phi_pad)
        I_cleft = self.model(phi_pad)
        if self.normalize is not None:
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


class CableSimulator:
    def __init__(self, const, coupling, Ord11_model_fn, device, dtype, dt=0.01, save_stride=100):
        self.const = const
        self.parameters = const.parameters
        self.coupling = coupling
        self.Ord11_model_fn = Ord11_model_fn
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

    def step(self, dt=None):
        if dt is not None:
            self.dt = float(dt)
        self.parameters["dt"] = self.dt

        I_couple = self.coupling.compute(self.phi_i, self.ti)

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

        self.ti += self.dt
        self.count += 1
        return I_couple
    def adaptive_step(self, ti):
        bcl = float(self.parameters["bcl"])
        if abs(math.fmod(ti, bcl)) < self.const.twin:
            return float(self.const.dt1), float(self.const.dt1_samp)
        else:
            return float(self.const.dt2), float(self.const.dt2_samp)
    def run(self, T=None, print_every=100.0):
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

                if print_every is not None and abs(math.fmod(self.ti, float(print_every))) < 1e-5:
                    print(f"time: {self.ti:.2f} ms, dt={dt:.4g}")

                I_couple = self.step(dt)
                if self.ti + 1e-12 >= self.next_sample_time:
                    self.phi_save.append(self.phi_i.clone())
                    self.I_save.append(I_couple.clone())
                    self.t_save.append(self.ti)
                    self.next_sample_time += dt_samp

        phi_save = torch.stack(self.phi_save, dim=0) if self.phi_save else torch.empty(0)
        I_save   = torch.stack(self.I_save, dim=0) if self.I_save else torch.empty(0)
        t_save   = torch.tensor(self.t_save, device=self.device, dtype=self.dtype) if self.t_save else torch.empty(0)

        print(f"Simulation time: {time.time() - start:.2f} seconds")
        return t_save, phi_save, I_save

