r"""Numba GPU backend for Faith's PD over a ``BPTree``.

Mirrors the CPU kernels in :mod:`._pd` on the device. Nothing here imports
``numba.cuda`` at module scope: both kernels are built inside
``_build_kernels(gpu)`` and refer only to the module they are handed, so the
same source compiles under ``numba.cuda`` on NVIDIA and ``numba.hip`` on AMD.

The device does both passes. The presence matrix goes up as one byte per
``(sample, taxon)``; the per-sample prefix -- four bytes per entry, and the
larger array of the two -- is built on the device and never crosses the bus.
That is the opposite of the obvious arrangement, and it is the whole reason
this engine can pay: scanning on the host would mean uploading the prefix
instead, quadrupling the transfer to save a kernel that costs far less than
the transfer does.

Both kernels run one block per sample, so the available parallelism is the
sample count. That suits the regime anyone would reach for a device in --
thousands of samples against tens of thousands of taxa -- and is a poor fit for
a handful of samples, where the CPU kernels win outright and should be used.
The reduction is a fixed shared-memory tree rather than ``atomic.add``, so a
given input gives bit-identical output run to run; atomics would have made
every parity assertion hostage to float64 non-associativity.

"""

# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

from time import perf_counter
from warnings import warn

import numpy as np

# Threads per block, for both kernels. Must be a power of two: the reduction
# halves it down to one, and the scan sweeps offsets up from one.
_TPB = 256

# Device memory the per-sample prefix may occupy, which sets the sample chunk.
_MAX_PREF_BYTES = 256 * 1024 * 1024

# Backends whose kernels failed to build or run in this process. Populated by
# _mark_gpu_unavailable so a failing compilation is not retried on every call.
_unavailable = set()

# backend name -> compiled kernel table, built once
_kernels = {}

# id(host array) -> device array, for the index arrays. _setup_pd_bp hands back
# freshly built lo/hi/lengths per call, but a caller looping over count matrices
# with one tree reuses them, so memoizing by identity turns those into one
# upload. The host arrays are kept alive in _keepalive because an id() is only
# meaningful for as long as its object is.
_device = {}
_keepalive = []


def _numba_gpu_module():
    """Return the Numba GPU module usable on this host, or None.

    Unlike :func:`skbio.stats.distance._gpu._numba_gpu_module_for`, there is no
    device-resident array to read a namespace from -- Faith's PD counts arrive
    as NumPy -- so the backends are probed directly.

    Returns
    -------
    module or None
        ``numba.cuda`` on an NVIDIA device, ``numba.hip`` on ROCm, or None when
        neither is installed and usable, or when this process has already given
        up on it. The caller then stays on the CPU kernels.

    """
    for want in ("cuda", "hip"):
        if want in _unavailable:
            continue
        try:
            mod = getattr(__import__("numba", fromlist=[want]), want)
        except Exception:
            continue
        try:
            if mod.is_available():
                return mod
        except Exception:
            continue
    return None


def _mark_gpu_unavailable(name):
    """Record that ``name``'s kernels cannot be used, warning once."""
    if name not in _unavailable:
        _unavailable.add(name)
        warn(
            f"The Numba GPU kernels could not be used for the '{name}' backend "
            "on this system; using the CPU kernels instead.",
            UserWarning,
        )


def _build_kernels(gpu):
    """Compile the two kernels for ``gpu`` (``numba.cuda`` or ``numba.hip``).

    Parameters
    ----------
    gpu : module
        The Numba GPU module to compile against.

    Returns
    -------
    dict
        ``{"scan": kernel, "reduce": kernel}``.

    """
    from numba import float64 as nb_f64, int32 as nb_i32

    @gpu.jit
    def _scan_k(presence, pref):  # pragma: no cover
        # runs on the device; coverage.py cannot instrument compiled PTX
        #
        # One block per sample. The block walks its row in _TPB-wide chunks,
        # doing a Hillis-Steele inclusive scan of each chunk in shared memory
        # and carrying a running total forward in a register, so a single block
        # scans a row of any length. Every thread computes the same carry from
        # the same shared slot, so no extra coordination is needed for it.
        s = gpu.blockIdx.x
        tid = gpu.threadIdx.x
        buf = gpu.shared.array(_TPB, nb_i32)

        n_taxa = presence.shape[1]
        if tid == 0:
            pref[s, 0] = 0

        carry = 0
        base = 0
        while base < n_taxa:
            j = base + tid
            v = 0
            if j < n_taxa and presence[s, j] != 0:
                v = 1
            buf[tid] = v
            gpu.syncthreads()

            off = 1
            while off < _TPB:
                add = 0
                if tid >= off:
                    add = buf[tid - off]
                gpu.syncthreads()
                if tid >= off:
                    buf[tid] = buf[tid] + add
                gpu.syncthreads()
                off *= 2

            if j < n_taxa:
                pref[s, j + 1] = carry + buf[tid]
            carry = carry + buf[_TPB - 1]
            # the carry read above must land before the next chunk overwrites buf
            gpu.syncthreads()
            base += _TPB

    @gpu.jit
    def _reduce_k(pref, lo, hi, lengths, out):  # pragma: no cover
        # One block per sample, threads striding the node axis, then a fixed
        # shared-memory tree reduction. Every node is independent -- its
        # descendant taxa are a contiguous run, so presence is one range test
        # against the prefix -- and the only coordination is the final sum.
        s = gpu.blockIdx.x
        tid = gpu.threadIdx.x
        buf = gpu.shared.array(_TPB, nb_f64)

        n_nodes = lengths.shape[0]
        acc = 0.0
        k = tid
        while k < n_nodes:
            if pref[s, hi[k]] - pref[s, lo[k]] > 0:
                acc += lengths[k]
            k += _TPB
        buf[tid] = acc
        gpu.syncthreads()

        stride = _TPB // 2
        while stride > 0:
            if tid < stride:
                buf[tid] += buf[tid + stride]
            gpu.syncthreads()
            stride //= 2
        if tid == 0:
            out[s] = buf[0]

    return {"scan": _scan_k, "reduce": _reduce_k}


def _upload(gpu, arr):
    """Return a device copy of ``arr``, reusing one made for the same object."""
    key = id(arr)
    dev = _device.get(key)
    if dev is None:
        dev = gpu.to_device(np.ascontiguousarray(arr))
        _device[key] = dev
        _keepalive.append(arr)
    return dev


def _reset():
    """Drop every cached device buffer. Used when a backend is abandoned."""
    _device.clear()
    _keepalive.clear()


def _prepare():
    """Resolve the GPU module and its kernels, or return ``(None, None)``."""
    gpu = _numba_gpu_module()
    if gpu is None:
        return None, None
    name = gpu.__name__.rsplit(".", 1)[-1]
    try:
        if name not in _kernels:
            _kernels[name] = _build_kernels(gpu)
    except Exception:
        _mark_gpu_unavailable(name)
        return None, None
    return gpu, _kernels[name]


def run_faith_pd_gpu(presence, lo, hi, lengths, out, timings=None):
    """Fill ``out`` with Faith's PD per sample, on the device.

    Parameters
    ----------
    presence : ndarray of uint8 of shape (n_samples, n_taxa)
        Present-taxon indicator, C-contiguous and in ascending tip order.
    lo, hi : ndarray of int32 of shape (n_nodes,)
        Half-open bounds of each node's descendant taxa in that column space.
    lengths : ndarray of float64 of shape (n_nodes,)
        Branch length of each node, in preorder.
    out : ndarray of float64 of shape (n_samples,)
        Output buffer, written in place.
    timings : dict, optional
        Diagnostic hook. When given, accumulates seconds spent in ``"upload"``,
        ``"scan"``, ``"reduce"`` and ``"download"``. Splitting the phases needs
        a device synchronize between each, so a timed run is slower than an
        untimed one and its total is not the figure to quote.

    Returns
    -------
    bool
        True if the device ran the computation. False means no device could
        serve the call and the caller should use the CPU kernels; ``out`` is
        then untouched.

    """
    gpu, kernels = _prepare()
    if gpu is None:
        return False
    name = gpu.__name__.rsplit(".", 1)[-1]

    def _tick(phase, t0):
        if timings is not None:
            gpu.synchronize()
            timings[phase] = timings.get(phase, 0.0) + perf_counter() - t0

    try:
        n_samples, n_taxa = presence.shape
        # The prefix is the larger of the two device arrays (four bytes per
        # entry against presence's one), so it sets the chunk.
        chunk = min(n_samples, max(1, _MAX_PREF_BYTES // ((n_taxa + 1) * 4)))

        t0 = perf_counter()
        d_lo, d_hi = _upload(gpu, lo), _upload(gpu, hi)
        d_len = _upload(gpu, lengths)
        _tick("upload", t0)

        d_pref = gpu.device_array((chunk, n_taxa + 1), dtype=np.int32)
        d_out = gpu.device_array(chunk, dtype=np.float64)

        for start in range(0, n_samples, chunk):
            stop = min(start + chunk, n_samples)
            rows = stop - start

            t0 = perf_counter()
            d_pres = gpu.to_device(np.ascontiguousarray(presence[start:stop]))
            _tick("upload", t0)

            t0 = perf_counter()
            kernels["scan"][rows, _TPB](d_pres, d_pref)
            _tick("scan", t0)

            t0 = perf_counter()
            kernels["reduce"][rows, _TPB](d_pref, d_lo, d_hi, d_len, d_out)
            _tick("reduce", t0)

            t0 = perf_counter()
            gpu.synchronize()
            out[start:stop] = d_out.copy_to_host()[:rows]
            _tick("download", t0)
    except Exception:
        # correctness first: a kernel that cannot build or run on this stack
        # must not fail the call, it must yield to the CPU path
        _mark_gpu_unavailable(name)
        _reset()
        return False
    return True
