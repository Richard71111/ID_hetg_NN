import torch
import math
import numpy as np
from Ionic_current.Ord11.initial import Initial_ORd11

class Constants:
    def __init__(
            self, 
            device,
            dtype=torch.float32,
            bcl: int = 400, 
            nbeats: int = 5,
            Ncell: int = 1,
            model: str = 'Ord11',
            GJ_coupling: float = 735,
            dt_factor: float = 1,
            stim_amplitude: float = 50
            ) -> None:
        
        # Basic parameters
        self.Nint = 1 # number of intracellular nodes
        self.dt_factor = dt_factor
        self.device = device
        self.dtype = dtype

        self.parameters = {}
        self.parameters['bcl'] = bcl
        self.parameters['nbeats'] = nbeats
        self.parameters['N'] = Ncell * self.Nint
        self.parameters['Ncell'] = Ncell
        self.parameters['celltype'] = 0 # 0: endo, 1: epi, 2: M cell
        # Cell geometry
        Cm        = 1e-8                   # membrane capacitance, uF/um^2
        L         = 100                    # cell length, um
        r         = 11                     # cell radius, um
        Aax       = 2 * torch.pi * r * L   # patch surface area, um^2
        Ad        = torch.pi * r**2        # disc surface area, um^2
        Atot      = 2 * Ad + Aax           # total surface area, um^2
        self.parameters['Ctot'] = Atot * Cm              # total capacitance, uF
        self.parameters['L']    = L
        self.parameters['r']    = r
        # Time parameters
        self.T    = bcl * nbeats                    # total simulation time, ms
        self.dt1  = 0.01 / self.dt_factor           # ms, dt between stim and twin (0.01 for EpC)
        self.dt2  = 0.1 / self.dt_factor            # ms, dt between twin and next stim (0.1 for EpC)
        self.dtS1 = self.dt1 / 5                    # ms, cleft concentration time step 1
        self.dtS2 = self.dt2 / 10                   # ms, cleft concentration time step 2
        self.Ns1 = np.round(self.dt1 / self.dtS1)   # operator splitting for cleft concentrations
        self.Ns2 = np.round(self.dt2 / self.dtS2)   # operator splitting for cleft concentrations
        self.dt1_samp = self.dt1*self.dt_factor*4        # ms, sampling interval
        self.dt2_samp = self.dt2*self.dt_factor*4        # ms, sampling interval
        self.twin     = 50.0                             # ms, time of peaks
        self.trange   = [0.0, self.T]                    # ms, time range of simulation
        self.ts       = self._get_time_variable()   # time sequence variable
        save_int      = bcl + 50                    # last x ms to save - save last cycle + 50ms before
        self.ts_save  = self.ts[self.ts >= (self.T - save_int)]  # time sequence variable to save
        # self.dt1      = tensor(self.dt1, device=self.device, dtype=self.dtype)
        # self.dt2      = tensor(self.dt2, device=self.device, dtype=self.dtype)
        # Concentration parameters
        Na_o = torch.full((self.parameters['N'],), 140.0, device=self.device, dtype=self.dtype)    # mM, extracellular Na+ concentration
        K_o  = torch.full((self.parameters['N'],), 5.4, device=self.device, dtype=self.dtype)      # mM, extracellular K+ concentration
        Ca_o = torch.full((self.parameters['N'],), 1.8, device=self.device, dtype=self.dtype)      # mM, extracellular Ca2+ concentration
        A_o  = Na_o + K_o + 2 * Ca_o  # mM, anion A- concentration, mM
        S = torch.cat(
            [Na_o, K_o, Ca_o, A_o], dim=0
        )
        self.S = S # concatenated concentration vector (4N, )
        # GJ coupling parameters
        self.parameters['Ggap'] = GJ_coupling # nS GJ conductance

        # Ionic model parameters
        self.model = 'Ord11'
        if model == 'Ord11':
            self.parameters['fSERCA'] = 1.0  # scaling factor for SERCA pump
            self.parameters['fRyR']   = 1.0  # scaling factor for RyR release
            self.parameters['ftauhL'] = 1.0  # scaling factor for L-type inactivation time constant
            self.parameters['fCaMKa'] = 1.0  # scaling factor for CaMK activation
            self.parameters['fIleak'] = 1.0  # scaling factor for SR leak current
            self.parameters['fJrel']  = 1.0  # scaling factor for RyR release flux

            self.parameters['Nstate'] = 41-1 # number of state variables in the model
            # stimulus parameters
            self.parameters['stim_dur'] = 2.0                # ms, stimulus duration
            self.parameters['stim_amp'] = stim_amplitude     # uA/uF, stimulus amplitude
            indstim = torch.zeros(self.parameters['N'], device=self.device, dtype=self.dtype)
            indstim[:3] = 1  # stimulate first three cells
            self.parameters['indstim'] = indstim # indicator vector for stimulated cells

        # Initial conditions
        self.x0 = Initial_ORd11(self.device, self.dtype)  # initial state variables
        self.phi0    = torch.full((self.parameters['N'],), self.x0[0].item(), device=self.device, dtype=self.dtype)  # initial membrane potential
        x0_values = self.x0[1:]                 # initial gating variables
        self.g0 = torch.repeat_interleave(x0_values, repeats=self.parameters['N']) # pyright: ignore[reportGeneralTypeIssues]
    def _get_time_variable(self):
        """Generate time sequence variable based on adaptive time step."""
        ti = self.trange[0]
        T_end = self.trange[1]
        ts = []

        while ti < T_end:
            if abs(math.fmod(ti, self.parameters['bcl'])) < self.twin:
                dt = self.dt1
                dt_samp = self.dt1_samp
            else:
                dt = self.dt2
                dt_samp = self.dt2_samp

            if abs(math.fmod(ti, dt_samp)) < 1e-8:
                ts.append(ti)

            ti = round(ti + dt, 5)

        return np.array(ts)


class Normalization:
    def __init__(self, scaler, device, slice, dtype=torch.float32) -> None:
        self.device  = device
        self.dtype   = dtype
        self.scaler  = scaler
        self.Vmean   = torch.tensor(self.scaler.mean_[2:], device=self.device, dtype=self.dtype)
        self.Vscale  = torch.tensor(self.scaler.scale_[2:], device=self.device, dtype=self.dtype)
        self.Imean   = torch.tensor(self.scaler.mean_[:2], device=self.device, dtype=self.dtype)
        self.Iscale  = torch.tensor(self.scaler.scale_[:2], device=self.device, dtype=self.dtype)
    def normalizeV(self, V):
        V_norm = (V - self.Vmean) / self.Vscale
        return V_norm
    def denormalizeV(self, V_norm):
        V = V_norm * self.Vscale + self.Vmean
        return V
    def normalizeI(self, I):
        I_norm = (I - self.Imean) / self.Iscale
        return I_norm
    def denormalizeI(self, I_norm):
        I = I_norm * self.Iscale + self.Imean
        return I
class NormalizationCNN:
    def __init__(self, stats, slices, device, dtype=torch.float32) -> None:
        self.Inmean = stats['Inmean'].to(device, dtype)[:, :, slices]
        self.Instd  = stats['Instd'].to(device, dtype)[:, :, slices]
        self.Outmean = stats['Outmean'].to(device, dtype)[:, :, slices]
        self.Outstd  = stats['Outstd'].to(device, dtype)[:, :, slices]

    def NormalizeInput(self, V):
        V_norm = (V - self.Inmean) / self.Instd
        return V_norm

    def DenormalizeInput(self, V_norm):
        V = V_norm * self.Instd + self.Inmean
        return V

    def NormalizeOutput(self, I):
        I_norm = (I - self.Outmean) / self.Outstd
        return I_norm

    def DenormalizeOutput(self, I_norm):
        I = I_norm * self.Outstd + self.Outmean
        return I

