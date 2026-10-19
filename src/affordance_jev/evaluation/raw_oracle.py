"""GT-only union oracle for overlapping binary proposals.

Pixels with identical proposal memberships are compressed to weighted atoms.
Mixed-integer Dinkelbach steps optimize an OR of whole masks. A time-limited
solve returns a feasible lower bound and a certified upper bound, never a
heuristic mislabeled as the exact oracle. No GT-dependent operation is applied
to the model or its saved proposals.
"""
import time
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def group_memberships(membership, weights):
    """Combine equal boolean rows, summing their class-count weights."""
    membership = np.asarray(membership, bool)
    packed = np.ascontiguousarray(np.packbits(membership, axis=1))
    keys = packed.view(np.dtype((np.void, packed.shape[1]))).reshape(-1)
    _, index, inverse = np.unique(keys, return_index=True, return_inverse=True)
    grouped = np.zeros((len(index), weights.shape[1]), np.int64)
    np.add.at(grouped, inverse, weights)
    return membership[index], grouped


def pixel_atoms(masks, gt, classes=8):
    assert masks.ndim == 3 and masks.shape[1:] == gt.shape
    valid = gt >= 0
    membership = masks[:, valid].T
    values = gt[valid].astype(np.int64)
    packed = np.ascontiguousarray(np.packbits(membership, axis=1))
    keys = packed.view(np.dtype((np.void, packed.shape[1]))).reshape(-1)
    _, index, inverse = np.unique(keys, return_index=True, return_inverse=True)
    counts = np.bincount(inverse * classes + values,
                         minlength=len(index) * classes).reshape(-1, classes)
    return membership[index], counts, np.bincount(values, minlength=classes)


def single_scores(membership, counts, class_id, gt_count):
    tp = membership.T @ counts[:, class_id]
    areas = membership.T @ counts.sum(1)
    return tp / (gt_count + areas - tp)


def union_oracle(membership, counts, class_id, gt_count, seconds=2., tolerance=1e-6):
    start = time.perf_counter()
    if gt_count <= 0:
        raise ValueError('Oracle is defined for a present target class')
    tp = counts[:, class_id]
    fp = counts.sum(1) - tp
    ids = np.flatnonzero(membership.T @ tp > 0)
    if not len(ids):
        return {'lower': 0., 'upper': 0., 'certified': True, 'chosen': [], 'solver_calls': 0,
                'seconds': time.perf_counter()-start, 'retained_for_solver': 0}
    # These are exact solver reductions, not proposal filtering at inference.
    candidate_keys = np.ascontiguousarray(np.packbits(membership[:, ids].T, axis=1))
    _, unique = np.unique(candidate_keys, axis=0, return_index=True)
    ids = ids[np.sort(unique)]
    active = membership[:, ids]
    pos, neg = active[tp > 0].astype(np.float32), active[fp > 0].astype(np.float32)
    pi, ni = pos.T @ pos, neg.T @ neg
    pc, nc = np.diag(pi), np.diag(ni)
    dominated = np.zeros(len(ids), bool)
    for i in range(len(ids)):
        for j in range(len(ids)):
            if i == j:
                continue
            if pi[i, j] == pc[i] and ni[i, j] == nc[j]:
                # j covers at least i's target pixels and no extra false pixels.
                strict = pc[j] > pc[i] or nc[j] < nc[i]
                if strict or ids[j] < ids[i]:
                    dominated[i] = True
                    break
    ids = ids[~dominated]
    active, hist = group_memberships(membership[:, ids], counts)
    keep = active.any(1)
    active, hist = active[keep], hist[keep]
    tp, fp = hist[:, class_id], hist.sum(1)-hist[:, class_id]
    n, atoms = len(ids), len(hist)
    recall_bound = float(tp.sum()/gt_count)

    def score(selected):
        cover = active[:, selected].any(1) if np.any(selected) else np.zeros(atoms, bool)
        return float(tp[cover].sum()/(gt_count+fp[cover].sum()))

    # Cheap feasible start; optimization below supplies the certificate.
    chosen = np.zeros(n, bool)
    lower = 0.
    for _ in range(2*n+1):
        best, value = None, lower
        for i in range(n):
            trial = chosen.copy(); trial[i] = ~trial[i]
            q = score(trial)
            if q > value + 1e-12:
                best, value = trial, q
        if best is None:
            break
        chosen, lower = best, value
    upper = recall_bound
    calls = 0
    last_status = None
    if upper-lower > tolerance:
        a, c = np.nonzero(active)
        edges = len(a)
        edge_row = np.arange(edges)
        rows = np.concatenate([edge_row, edge_row, edges+np.arange(atoms), edges+a])
        cols = np.concatenate([c, n+a, n+np.arange(atoms), c])
        data = np.concatenate([np.ones(edges), -np.ones(edges), np.ones(atoms), -np.ones(edges)])
        matrix = coo_matrix((data,(rows,cols)),shape=(edges+atoms,n+atoms)).tocsc()
        constraints = LinearConstraint(matrix, -np.inf, 0.)
        integrality = np.r_[np.ones(n), np.zeros(atoms)]
        while upper-lower > tolerance and calls < 12:
            remaining = seconds-(time.perf_counter()-start)
            if remaining <= 0:
                break
            ratio = lower
            objective = np.r_[np.zeros(n), (ratio*fp-tp)/gt_count]
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Unrecognized options detected')
                solved = milp(objective, integrality=integrality, bounds=Bounds(0,1),
                    constraints=constraints, options={'time_limit':remaining, 'mip_rel_gap':1e-9,
                                                       'presolve':True, 'threads':1})
            calls += 1
            last_status = int(solved.status)
            if solved.x is not None:
                trial = solved.x[:n] > .5
                value = score(trial)
                if value > lower:
                    chosen, lower = trial, value
            dual = getattr(solved, 'mip_dual_bound', None)
            if dual is not None and np.isfinite(dual):
                # IoU(S) <= ratio + max(0, max_S(TP-ratio*FP)/G-ratio).
                bound = ratio + max(0., -float(dual)-ratio) + 1e-8
                upper = min(upper, max(lower, bound))
            if solved.status not in [0,1] or (solved.x is None and dual is None):
                break
    assert lower <= upper+1e-7
    return {'lower':lower, 'upper':max(lower,upper), 'certified':upper-lower <= tolerance,
            'chosen':ids[chosen].tolist(), 'solver_calls':calls, 'last_solver_status':last_status,
            'seconds':time.perf_counter()-start, 'retained_for_solver':n}
