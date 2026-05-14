HPC reproducibility guide
=========================

End-to-end recipe for running the global ICON+ETOPO CSA pipeline:
download the input data, pick a hardware preset, submit (locally or
via SLURM), monitor, recover from a crash, and validate the output.

This page consolidates what used to live in five untracked Markdown
files (``runs/HPC_ETOPO_README.md``, ``runs/QUICK_START_128_CORES.md``,
``runs/NETCDF_CHUNKING_GUIDE.md``, ``runs/README_CHUNKING.md``,
``scripts/ETOPO_DOWNLOAD_GUIDE.md``) — those have been removed.

.. contents::
   :local:
   :depth: 2


Overview
--------

The global pipeline processes all 20,480 cells of the ICON R02B04 grid.
For each land cell it loads the bounding ETOPO 15″ tile(s), runs the
CSA approximation, and writes per-cell spectra into NetCDF chunks (100
cells per file by default → ~205 files). Ocean cells are detected at
load-time and skipped cheaply.

Expected runtime depends on the chosen ``SYSTEM_CONFIG`` (see below);
the DKRZ HPC preset finishes ~20 k land cells in roughly 6–8 hours.


Hardware sizing
---------------

Memory is the limiting factor. Per-cell footprint scales with latitude:
equatorial cells fit in ~10 GB, polar cells (high lon-density in
degree-space) can need 25–60 GB. The pipeline groups cells into
*memory batches* (see :ref:`architecture`) so the worker count adapts
batch-to-batch.

The presets in ``runs/icon_etopo_global.py`` cover three target
machines:

.. code-block:: text

   generic_laptop       12 cores / 12 GB / chunks of 100 cells
   laptop_performance   20 cores / 80 GB / chunks of 100 cells
   dkrz_hpc             250 cores / 240 GB / chunks of 100 cells

To add a new preset, edit the ``CONFIGS`` dict near line 479 of
``runs/icon_etopo_global.py``.


Download ETOPO 2022 (15 arc-second)
-----------------------------------

The pipeline uses **surface** elevation (ice surface where ice exists,
bedrock elsewhere) — that's the global product. The *bedrock-only*
product exists only for polar regions and isn't what this pipeline
expects.

* Source: NOAA NCEI, DOI ``10.25921/fd45-gt74``.
* Resolution: 15″ (~450 m at equator).
* Coverage: global; 288 tiles at 15° × 15° each.
* Total dataset: ~50–100 GB.

Single tile (example: 45°N, 120°W — US west coast):

.. code-block:: bash

   BASE=https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/15s/15s_surface_elev_netcdf
   wget "$BASE/ETOPO_2022_v1_15s_N45W120_surface.nc"

Full global download (run from a SLURM job — bandwidth-limited):

.. code-block:: bash

   BASE=https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/15s/15s_surface_elev_netcdf
   OUT=$HOME/data/etopo_15s
   mkdir -p "$OUT"
   for lat in N90 N75 N60 N45 N30 N15 N00 S15 S30 S45 S60 S75; do
       for lon in W180 W165 W150 W135 W120 W105 W090 W075 W060 W045 W030 W015 \
                  E000 E015 E030 E045 E060 E075 E090 E105 E120 E135 E150 E165; do
           wget -c -P "$OUT" "$BASE/ETOPO_2022_v1_15s_${lat}${lon}_surface.nc"
       done
   done

Verify a file::

   ncdump -h $HOME/data/etopo_15s/ETOPO_2022_v1_15s_N45W120_surface.nc

Then point ``pycsa.local_paths.paths.etopo`` (or your edited config) at
``$HOME/data/etopo_15s/``.


Configure
---------

Open ``runs/icon_etopo_global.py`` and set two things near the top of
the ``if __name__ == "__main__"`` block:

1. **System preset** (around line 456)::

       SYSTEM_CONFIG = "dkrz_hpc"   # or "laptop_performance" / "generic_laptop"

2. **Cell range** (around line 646)::

       cell_start = 0          # first cell, inclusive
       cell_end   = None       # last cell, exclusive; None = run to the end

To regenerate a single chunk (cells 2900–2999) after a crash::

       cell_start = 2900
       cell_end   = 3000


Run
---

Direct invocation (on the target node)::

   python -m runs.icon_etopo_global

SLURM (recommended on a cluster — see the existing
``runs/submit_etopo_global.sh`` for memory/time limits)::

   sbatch runs/submit_etopo_global.sh
   squeue -u $USER
   tail -f logs/icon_etopo_global_*.log


Monitor
-------

Dask dashboard URL is logged on startup (typically
``http://127.0.0.1:8787/status``). Tunnel from a workstation if running
on a remote node:

.. code-block:: bash

   ssh -L 8787:localhost:8787 user@hpc-node
   # then open http://localhost:8787

Progress via the filesystem:

.. code-block:: bash

   # completed NetCDF chunks
   ls outputs/global_run/datasets/icon_etopo_global_cells_*.nc | wc -l

   # diagnostic plots written so far
   find outputs/icon_etopo_global -name 'cell_*.png' | wc -l


Restart after a crash
---------------------

The pipeline writes one NetCDF chunk per ``netcdf_chunk_size`` cells
and processes memory batches in latitude order (equatorial → mid-lat →
polar). After a crash the completed chunks are on disk; the
``cell_start`` knob in the script picks the resume point.

.. code-block:: bash

   # See which chunks finished
   ls outputs/global_run/datasets/icon_etopo_global_cells_*.nc

   # Identify the first incomplete chunk (e.g. cells 12000-12099 is missing)
   # Edit runs/icon_etopo_global.py:
   #   cell_start = 12000
   #   cell_end   = None
   # then re-submit.

Each ``do_cell`` call is wrapped in ``try/except`` that logs the
traceback before re-raising, so a single bad cell surfaces a stack
instead of hanging the whole batch.


.. _architecture:

Architecture
------------

**Dual-loop chunking** separates Dask parallelism from on-disk file
organisation:

* *Processing batch* — how many cells get submitted to the Dask client
  in one ``client.compute(...)`` call. Sized to keep all workers busy.
* *NetCDF chunk* — how many cells go into one output file. Independent
  of parallelism; sized for crash recovery and manageability. All
  current presets use ``netcdf_chunk_size = 100``.

**Memory batching** groups cells by estimated memory requirement
(``pycsa.scheduling.group_cells_by_memory``, latitude-driven). Each
memory batch is processed under its own Dask client with worker count
sized to fit the batch's per-cell memory inside the node's total RAM.
Polar batches end up with fewer, larger-memory workers; equatorial
batches with more, smaller-memory workers.

**Tile cache** — :mod:`pycsa.core.tile_cache` holds a per-Dask-worker
singleton (``_WORKER_CACHE``) initialised at the start of each memory
batch via ``client.run(init_worker_cache, ...)``. ``do_cell`` retrieves
it via ``ctx.tile_cache()`` (see :class:`pycsa.compute.ComputeContext`)
so the ETOPO tile handles stay open across cells in the same worker
instead of being re-opened per cell.


Post-processing
---------------

After the run finishes (or whenever you want to consolidate)::

   # Sanity-check chunk coverage (no gaps, expected count)
   python -m runs.validate_chunks

   # Merge all chunks into one NetCDF file
   python -m runs.merge_netcdf_chunks
   # Optional: --cleanup to delete the per-chunk files after merging
   # Optional: --output icon_etopo_global_v2.nc to rename

   # Cross-check land/ocean ratio against an ETOPO-derived reference
   python scripts/verify_icon_etopo_land_ocean.py

The merged file lives at
``outputs/global_run/datasets/icon_etopo_global_FINAL.nc``
(~1.8 GB for a full global run).


Output layout
-------------

::

   outputs/
   ├── icon_etopo_global/                    # diagnostic per-cell plots
   │   ├── cells_00000-00099/
   │   │   ├── cell_00000.png
   │   │   └── ...
   │   └── cells_00100-00199/
   └── global_run/datasets/                  # spectral NetCDF chunks
       ├── icon_etopo_global_cells_00000-00099.nc
       ├── icon_etopo_global_cells_00100-00199.nc
       └── ...
                                              # (+ icon_etopo_global_FINAL.nc
                                              #     after merge)


Troubleshooting
---------------

**OOM / "KilledWorker" in logs.** The memory batch underestimated
per-cell footprint. Either raise the safety factor in
``pycsa.scheduling.group_cells_by_memory`` (default 1.0) or pick a
preset with more memory per worker.

**"Too many open files."** Raise the descriptor limit before
launching::

   ulimit -n 4096

**Dask processes won't terminate cleanly.** The tile cache holds
NetCDF file handles for the worker lifetime; ``init_worker_cache`` is
idempotent and the cache is freed when the worker process exits, so
killing the client and letting workers shut down is the supported
recovery path.

**Chunks have gaps.** Some cells failed. Run ``python -m
runs.validate_chunks`` to list missing ranges, then re-run with
``cell_start`` / ``cell_end`` covering the gaps.


Citation
--------

When you use ETOPO 2022 data, cite::

   NOAA National Centers for Environmental Information. 2022:
   ETOPO 2022 15 Arc-Second Global Relief Model. NOAA NCEI.
   https://doi.org/10.25921/fd45-gt74

The pyCSA software citation is in the ``CITATION.cff`` at the repo root
(GitHub renders a *"Cite this repository"* button on the project page).
