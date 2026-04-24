from ase.calculators.calculator import Calculator, all_changes
from ase.data import chemical_symbols
from aimnet2calc import AIMNet2Calculator
from typing import Union
import itertools
import torch
import numpy as np


# unit conversion for remote mode (server returns pysisyphus a.u., ASE wants eV)
_AU2EV    = 27.211386245988
_ANG2BOHR = 1.8897261245650618
_BOHR2ANG = 1.0 / _ANG2BOHR


class AIMNet2ASE(Calculator):
    implemented_properties = ['energy', 'forces', 'free_energy', 'charges', 'stress']

    # Remote-mode class attributes.  When `enable_remote` has been called,
    # `calculate()` forwards (atoms, coords) to the shared BatchGPUServer
    # via multiprocessing queues instead of running a local AIMNet2 forward.
    _remote_request_queue = None
    _remote_response_queue = None
    _remote_worker_id = None
    _remote_counter = itertools.count()

    @classmethod
    def enable_remote(cls, request_queue, response_queue, worker_id):
        """Route all calculator calls to a BatchGPUServer in another process."""
        cls._remote_request_queue = request_queue
        cls._remote_response_queue = response_queue
        cls._remote_worker_id = worker_id

    @classmethod
    def _is_remote(cls):
        return cls._remote_request_queue is not None

    def __init__(self, base_calc: Union[AIMNet2Calculator, str] = 'aimnet2', charge=0, mult=1):
        super().__init__()
        if AIMNet2ASE._is_remote():
            # no local model — requests go through the GPU server
            self.base_calc = None
            self.implemented_species = None
            self.charge = charge
            self.mult = mult
            # still null out the cached tensors so that reset() / set_charge /
            # set_mult don't crash if called
            self._t_numbers = None
            self._t_charge = None
            self._t_mult = None
            self._t_mol_idx = None
            return
        if isinstance(base_calc, str):
            base_calc = AIMNet2Calculator(base_calc)
        self.base_calc = base_calc
        self.charge = charge
        self.mult = mult
        self.reset()
        # list of implemented species
        if hasattr(base_calc, 'implemented_species'):
            self.implemented_species = base_calc.implemented_species.cpu().numpy()
        else:
            self.implemented_species = None

    def reset(self):
        super().reset()
        self._t_numbers = None
        self._t_charge = None
        self._t_mult = None
        self._t_mol_idx = None
        self.charge = 0.0
        self.mult = 1.0

    def set_atoms(self, atoms):
        if self.implemented_species is not None and not np.in1d(atoms.numbers, self.implemented_species).all():
            raise ValueError('Some species are not implemented in the AIMNet2Calculator')
        self.reset()
        self.atoms = atoms

    def set_charge(self, charge):
        self.charge = charge
        self._t_charge = None
        self.update_tensors()

    def set_mult(self, mult):
        self.mult = mult
        self._t_mult = None
        self.update_tensors()

    def update_tensors(self):
        if AIMNet2ASE._is_remote():
            return   # no tensors to build; inputs go through the queue
        if self._t_numbers is None:
            self._t_numbers = torch.tensor(self.atoms.numbers, dtype=torch.int64, device=self.base_calc.device)
        if self._t_charge is None:
            self._t_charge = torch.tensor(self.charge, dtype=torch.float32, device=self.base_calc.device)
        if self._t_mult is None:
            self._t_mult = torch.tensor(self.mult, dtype=torch.float32, device=self.base_calc.device)
        if self._t_mol_idx is None:
            self.mol_idx = torch.zeros(len(self.atoms), dtype=torch.int64, device=self.base_calc.device)

    def _calculate_remote(self, properties):
        """Dispatch a forces request to the BatchGPUServer (pysis protocol:
        Bohr in, Hartree / Hartree·Bohr⁻¹ out).  ASE's native units are
        Å / eV / eV·Å⁻¹, so we convert at both ends.  stress is not
        supported in remote mode."""
        coords_bohr = (self.atoms.positions * _ANG2BOHR).flatten().astype(np.float64)
        atom_syms = [chemical_symbols[n] for n in self.atoms.numbers]
        req_id = next(AIMNet2ASE._remote_counter)
        worker_id = AIMNet2ASE._remote_worker_id
        req = (worker_id, req_id, 'forces', atom_syms, coords_bohr,
               float(self.charge), float(self.mult))
        AIMNet2ASE._remote_request_queue.put(req)
        while True:
            resp_id, result = AIMNet2ASE._remote_response_queue.get()
            if resp_id == req_id:
                if isinstance(result, tuple) and result and result[0] == '__error__':
                    raise RuntimeError(result[1])
                break
            # not ours; put it back (shouldn't happen with per-worker queue)
            AIMNet2ASE._remote_response_queue.put((resp_id, result))
        # convert a.u. → ASE (eV, eV/Å)
        energy_eV = float(result['energy']) * _AU2EV
        forces_evang = (
            np.asarray(result['forces']).reshape(-1, 3) * (_AU2EV / _BOHR2ANG)
        )
        self.results['energy']  = energy_eV
        self.results['charges'] = np.zeros(len(self.atoms))
        if 'forces' in properties:
            self.results['forces'] = forces_evang

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if AIMNet2ASE._is_remote():
            return self._calculate_remote(properties)
        self.update_tensors()

        if self.atoms.cell is not None and self.atoms.pbc.any():
            #assert self.base_calc.cutoff_lr < float('inf'), 'Long-range cutoff must be finite for PBC'
            cell = self.atoms.cell.array
        else:
            cell = None

        results = self.base_calc({
            'coord': torch.tensor(self.atoms.positions, dtype=torch.float32, device=self.base_calc.device),
            'numbers': self._t_numbers,
            'cell': cell,
            'mol_idx': self._t_mol_idx,
            'charge': self._t_charge,
            'mult': self._t_mult,
        }, forces='forces' in properties, stress='stress' in properties)
        for k, v in results.items():
            results[k] = v.detach().cpu().numpy()

        self.results['energy'] = results['energy']
        self.results['charges'] = results['charges']
        if 'forces' in properties:
            self.results['forces'] = results['forces']
        if 'stress' in properties:
            self.results['stress'] = results['stress']
