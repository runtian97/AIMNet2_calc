from pysisyphus.calculators.Calculator import Calculator
from pysisyphus.elem_data import ATOMIC_NUMBERS
from pysisyphus.constants import BOHR2ANG, ANG2BOHR, AU2EV
from aimnet2calc import AIMNet2Calculator
from typing import Dict, List, Union
import itertools
import numpy as np
import torch


EV2AU = 1 / AU2EV


class AIMNet2Pysis(Calculator):
    implemented_properties = ['energy', 'forces', 'free_energy', 'charges', 'stress']
    # Shared model cache: model path/key -> AIMNet2Calculator instance
    _model_cache: Dict[str, AIMNet2Calculator] = {}

    # Remote-mode class attributes.  When set (via enable_remote), get_energy /
    # get_forces / get_hessian forward their inputs to a BatchGPUServer via
    # multiprocessing queues instead of calling the GPU locally.
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

    def __init__(self, model: Union[AIMNet2Calculator, str] = 'aimnet2', charge=0, mult=1, **kwargs):
        super().__init__(charge=charge, mult=mult, **kwargs)
        if AIMNet2Pysis._is_remote():
            # No local model — requests go through the GPU server.
            self.model = None
            return
        if isinstance(model, str):
            if model not in AIMNet2Pysis._model_cache:
                AIMNet2Pysis._model_cache[model] = AIMNet2Calculator(model)
            self.model = AIMNet2Pysis._model_cache[model]
        elif isinstance(model, AIMNet2Calculator):
            self.model = model
        else:
            raise ValueError(f'Invalid model type: {type(model)}')

    def _remote_call(self, req_type, atoms, coords):
        """Send a request to the BatchGPUServer and block for the result."""
        req_id = next(AIMNet2Pysis._remote_counter)
        worker_id = AIMNet2Pysis._remote_worker_id
        coords_np = np.asarray(coords, dtype=np.float64)
        req = (worker_id, req_id, req_type, list(atoms), coords_np,
               float(self.charge), float(self.mult))
        AIMNet2Pysis._remote_request_queue.put(req)
        while True:
            resp_id, result = AIMNet2Pysis._remote_response_queue.get()
            if resp_id == req_id:
                if isinstance(result, tuple) and result and result[0] == '__error__':
                    raise RuntimeError(result[1])
                return result
            else:
                # Should not happen with per-worker response queues; defensive.
                AIMNet2Pysis._remote_response_queue.put((resp_id, result))

    def _prepere_input(self, atoms, coord):
        device = self.model.device
        numbers = torch.as_tensor([ATOMIC_NUMBERS[a.lower()] for a in atoms], device=device)
        coord = torch.as_tensor(coord, dtype=torch.float, device=device).view(-1, 3) * BOHR2ANG
        charge = torch.as_tensor([self.charge], dtype=torch.float, device=device)
        mult = torch.as_tensor([self.mult], dtype=torch.float, device=device)
        return dict(coord=coord, numbers=numbers, charge=charge, mult=mult)

    @staticmethod
    def _results_get_energy(results):
        return results['energy'].item() * EV2AU

    @staticmethod
    def _results_get_forces(results):
        return (results['forces'].detach() * (EV2AU / ANG2BOHR)).flatten().to(torch.double).cpu().numpy()

    @staticmethod
    def _results_get_hessian(results):
        return (results['hessian'].flatten(0, 1).flatten(-2, -1) * (EV2AU / ANG2BOHR / ANG2BOHR)).to(torch.double).cpu().numpy()

    def get_energy(self, atoms, coords):
        if AIMNet2Pysis._is_remote():
            r = self._remote_call('forces', atoms, coords)
            return dict(energy=r['energy'])
        _in = self._prepere_input(atoms, coords)
        res = self.model(_in)
        energy = self._results_get_energy(res)
        return dict(energy=energy)

    def get_forces(self, atoms, coords):
        if AIMNet2Pysis._is_remote():
            return self._remote_call('forces', atoms, coords)
        _in = self._prepere_input(atoms, coords)
        res = self.model(_in, forces=True)
        energy = self._results_get_energy(res)
        forces = self._results_get_forces(res)
        return dict(energy=energy, forces=forces)

    def get_hessian(self, atoms, coords):
        if AIMNet2Pysis._is_remote():
            return self._remote_call('hessian', atoms, coords)
        _in = self._prepere_input(atoms, coords)
        res = self.model(_in, forces=True, hessian=True)
        energy = self._results_get_energy(res)
        forces = self._results_get_forces(res)
        hessian = self._results_get_hessian(res)
        return dict(energy=energy, forces=forces, hessian=hessian)

    def _build_batch_input(self, atoms_list, coords_list):
        """Pack per-molecule atoms/coords into (B, N, 3) tensors.

        All molecules must share the same atom count N. Atom types may differ
        across molecules (cross-reaction batching).
        coords_list in Bohr (pysisyphus convention).
        """
        B = len(atoms_list)
        assert B > 0, "Empty batch"
        device = self.model.device
        nat = len(atoms_list[0])
        assert all(len(a) == nat for a in atoms_list), \
            "All molecules in a batch must have the same atom count"

        # (B, N) numbers — per-molecule atomic numbers
        numbers = torch.stack([
            torch.as_tensor(
                [ATOMIC_NUMBERS[a.lower()] for a in atoms],
                dtype=torch.int, device=device,
            )
            for atoms in atoms_list
        ])

        # (B, N, 3) coords — Bohr → Å
        coords_tensor = torch.stack([
            torch.as_tensor(c, dtype=torch.float, device=device).view(nat, 3) * BOHR2ANG
            for c in coords_list
        ])

        charge = torch.full((B,), self.charge, dtype=torch.float, device=device)
        mult   = torch.full((B,), self.mult,   dtype=torch.float, device=device)

        return dict(coord=coords_tensor, numbers=numbers, charge=charge, mult=mult), B, nat

    def _remote_call_batch(self, req_type, atoms_list, coords_list):
        """Send a batch of requests to the BatchGPUServer and collect results."""
        worker_id = AIMNet2Pysis._remote_worker_id
        req_ids = []
        for atoms, coords in zip(atoms_list, coords_list):
            req_id = next(AIMNet2Pysis._remote_counter)
            coords_np = np.asarray(coords, dtype=np.float64)
            req = (worker_id, req_id, req_type, list(atoms), coords_np,
                   float(self.charge), float(self.mult))
            AIMNet2Pysis._remote_request_queue.put(req)
            req_ids.append(req_id)

        pending = set(req_ids)
        results = {}
        while pending:
            resp_id, result = AIMNet2Pysis._remote_response_queue.get()
            if resp_id in pending:
                if isinstance(result, tuple) and result and result[0] == '__error__':
                    raise RuntimeError(result[1])
                results[resp_id] = result
                pending.discard(resp_id)
            else:
                AIMNet2Pysis._remote_response_queue.put((resp_id, result))
        return [results[rid] for rid in req_ids]

    def batch_get_forces(self, atoms_list: List[List[str]], coords_list: List) -> List[dict]:
        """Evaluate energy+forces for B geometries in a single GPU batch.

        All B geometries must share the same atom count. Atom types can differ
        across molecules. Returns list of {'energy', 'forces'} in a.u.
        """
        if AIMNet2Pysis._is_remote():
            return self._remote_call_batch('forces', atoms_list, coords_list)
        _in, B, _ = self._build_batch_input(atoms_list, coords_list)
        res = self.model(_in, forces=True)
        results_list = []
        for i in range(B):
            energy = res['energy'][i].item() * EV2AU
            forces = (
                res['forces'][i].detach() * (EV2AU / ANG2BOHR)
            ).flatten().to(torch.double).cpu().numpy()
            results_list.append(dict(energy=energy, forces=forces))
        return results_list

    def batch_get_hessian(self, atoms_list: List[List[str]], coords_list: List) -> List[dict]:
        """Evaluate energy+forces+hessian for B geometries in a single GPU batch.

        Bypasses AIMNet2Calculator's explicit `mol_idx[-1] > 0` rejection —
        verified that the underlying calculate_hessian produces correct
        per-molecule blocks (cross-molecule blocks are zero because the
        neighbour list respects mol_idx).

        All B geometries must share the same atom count. Atom types can differ.
        Returns list of {'energy', 'forces', 'hessian'} in a.u.
        """
        if AIMNet2Pysis._is_remote():
            return self._remote_call_batch('hessian', atoms_list, coords_list)
        _in, B, nat = self._build_batch_input(atoms_list, coords_list)

        calc = self.model   # AIMNet2Calculator instance
        data = calc.prepare_input(_in)
        # SKIP the `if hessian and data['mol_idx'][-1] > 0: raise` check
        data = calc.set_grad_tensors(data, forces=True, stress=False, hessian=True)
        with torch.jit.optimized_execution(False):
            data = calc.model(data)  # inner torch.jit model
        data = calc.get_derivatives(data, forces=True, stress=False, hessian=True)
        data = calc.process_output(data)
        # After process_output: energy (B,), forces (B, N, 3), hessian (B*N, 3, B*N, 3)

        # Extract per-molecule hessian blocks. Cross-molecule blocks are zero.
        hess_full = data['hessian']                        # (B*N, 3, B*N, 3)
        hess_5d   = hess_full.view(B, nat, 3, B, nat, 3)
        # block-diagonal: take hess_5d[b, :, :, b, :, :] for each b
        hess_per_mol = torch.stack([hess_5d[b, :, :, b, :, :] for b in range(B)])  # (B, N, 3, N, 3)

        results_list = []
        for i in range(B):
            energy = data['energy'][i].item() * EV2AU
            forces = (
                data['forces'][i].detach() * (EV2AU / ANG2BOHR)
            ).flatten().to(torch.double).cpu().numpy()
            hessian = (
                hess_per_mol[i].detach() * (EV2AU / ANG2BOHR / ANG2BOHR)
            ).flatten(0, 1).flatten(-2, -1).to(torch.double).cpu().numpy()  # (3N, 3N)
            results_list.append(dict(energy=energy, forces=forces, hessian=hessian))

        return results_list


def _patch_cos_for_aimnet_batch():
    """Monkey-patch ChainOfStates.calculate_forces to use AIMNet2 batch evaluation.

    When all COS images share the same AIMNet2Calculator model instance, their forces
    are computed in a single batched GPU forward pass instead of one at a time.
    Compatible with pysisyphus 0.8.x (calculate_forces, self.scheduler, self.counter).
    """
    from pysisyphus.cos import ChainOfStates as _cos_mod
    import sys

    ChainOfStates = _cos_mod.ChainOfStates
    _original = ChainOfStates.calculate_forces

    def _batch_calculate_forces(self):
        # replicate the image selection logic from the original
        images_to_calculate = self.moving_images
        if self.fix_first and (self.images[0]._energy is None):
            images_to_calculate = [self.images[0]] + images_to_calculate
        if self.fix_last and (self.images[-1]._energy is None):
            images_to_calculate = images_to_calculate + [self.images[-1]]

        if not self.scheduler and len(images_to_calculate) > 0:
            calcs = [img.calculator for img in images_to_calculate]
            nat0 = len(images_to_calculate[0].atoms)
            can_batch = (
                all(isinstance(c, AIMNet2Pysis) for c in calcs)
                and len({id(c.model) for c in calcs}) == 1
                and all(len(img.atoms) == nat0 for img in images_to_calculate)
                and all(c.charge == calcs[0].charge and c.mult == calcs[0].mult for c in calcs)
            )
            if can_batch:
                print(f"[AIMNet2 batch] {len(images_to_calculate)} images → 1 forward pass", flush=True)
                atoms_list = [list(img.atoms) for img in images_to_calculate]
                coords_list = [img.cart_coords for img in images_to_calculate]
                results_list = calcs[0].batch_get_forces(atoms_list, coords_list)
                for img, results in zip(images_to_calculate, results_list):
                    img.set_results(results)
                if self.progress:
                    print("." * len(images_to_calculate) + "\r", end="")
                    sys.stdout.flush()
                self.set_zero_forces_for_fixed_images()
                self.counter += 1
                energies = [image.energy for image in self.images]
                forces = np.array([image.forces for image in self.images])
                self.all_energies.append(energies)
                self.all_true_forces.append(forces)
                return {"energies": energies, "forces": forces}

        return _original(self)

    ChainOfStates.calculate_forces = _batch_calculate_forces




# ── Cross-reaction GPU batching ───────────────────────────────────────────────
class BatchGPUServer:
    """Collects force/hessian requests from multiple worker subprocesses and
    serves them as batched AIMNet2 forward passes, grouped by (nat, charge,
    mult, req_type).

    Runs in the main process.  Workers (subprocesses) send requests via a
    shared multiprocessing.Queue and receive results on per-worker response
    queues.  Within each flush window (default 5 ms) the server drains the
    request queue, groups compatible requests, and issues one GPU call per
    group.

    Request tuple:
        (worker_id, req_id, req_type, atoms_list, coords_np, charge, mult)

    Response tuple:
        (req_id, result_dict)  or  (req_id, ('__error__', message))
    """

    def __init__(self, model, flush_timeout_s: float = 0.005):
        import multiprocessing as mp
        self._model_arg = model
        self._flush_timeout_s = flush_timeout_s
        self._mp_ctx = mp.get_context('spawn')
        self.request_queue = self._mp_ctx.Queue()
        self.response_queues: Dict[int, 'mp.Queue'] = {}
        self._pysis_pool: Dict[tuple, AIMNet2Pysis] = {}
        self._thread = None
        self._stop_event = None
        self._stats = dict(batches=0, requests=0)

    def register_worker(self, worker_id: int):
        """Allocate a response queue for the given worker.  Must be called
        before the worker process is spawned (so the queue can be pickled
        into it)."""
        assert worker_id not in self.response_queues
        self.response_queues[worker_id] = self._mp_ctx.Queue()
        return self.request_queue, self.response_queues[worker_id]

    def start(self):
        import threading
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name='BatchGPUServer')
        self._thread.start()

    def stop(self, timeout: float = 5.0):
        """Signal the worker thread to shut down and wait for it to finish."""
        if self._stop_event is not None:
            self._stop_event.set()
        self.request_queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        print(f"[BatchGPUServer] served {self._stats['requests']} requests in "
              f"{self._stats['batches']} batches", flush=True)

    def _get_pysis(self, charge: float, mult: float) -> 'AIMNet2Pysis':
        key = (charge, mult)
        if key not in self._pysis_pool:
            self._pysis_pool[key] = AIMNet2Pysis(
                model=self._model_arg, charge=charge, mult=mult,
            )
        return self._pysis_pool[key]

    def _loop(self):
        from queue import Empty
        import time
        from collections import defaultdict
        print(f"[BatchGPUServer] ready (flush={self._flush_timeout_s*1000:.1f} ms)", flush=True)
        while not self._stop_event.is_set():
            try:
                first = self.request_queue.get()
            except (EOFError, BrokenPipeError):
                break
            if first is None:
                break
            pending = [first]
            # Drain additional requests that arrive within the flush window
            deadline = time.time() + self._flush_timeout_s
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    req = self.request_queue.get(timeout=remaining)
                except Empty:
                    break
                if req is None:
                    self._process(pending)
                    return
                pending.append(req)
            self._process(pending)

    def _process(self, pending: list):
        from collections import defaultdict
        groups = defaultdict(list)
        for req in pending:
            worker_id, req_id, req_type, atoms, coords, charge, mult = req
            key = (len(atoms), float(charge), float(mult), req_type)
            groups[key].append(req)

        for (nat, charge, mult, req_type), reqs in groups.items():
            calc = self._get_pysis(charge, mult)
            atoms_list  = [r[3] for r in reqs]
            coords_list = [r[4] for r in reqs]
            try:
                if req_type == 'forces':
                    results = calc.batch_get_forces(atoms_list, coords_list)
                elif req_type == 'hessian':
                    results = calc.batch_get_hessian(atoms_list, coords_list)
                else:
                    raise ValueError(f"Unknown req_type: {req_type}")
                print(f"[BatchGPU] {req_type} B={len(reqs)} nat={nat}", flush=True)
                self._stats['batches'] += 1
                self._stats['requests'] += len(reqs)
                for req, result in zip(reqs, results):
                    worker_id, req_id = req[0], req[1]
                    self.response_queues[worker_id].put((req_id, result))
            except Exception as e:
                import traceback
                traceback.print_exc()
                msg = f"{type(e).__name__}: {e}"
                for req in reqs:
                    worker_id, req_id = req[0], req[1]
                    self.response_queues[worker_id].put((req_id, ('__error__', msg)))
