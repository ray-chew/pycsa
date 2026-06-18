<p align="center">
  <a href="https://ray-chew.github.io/pyCSA/index.html">
  <img alt="CSA Logo" src="https://ray-chew.github.io/pyCSA/_static/logo.png" width="180">
  </a>
</p>

<h2 align="center">Constrained Spectral Approximation</h2>


<p align="center">
<a href="https://github.com/ray-chew/pyCSA/actions/workflows/ci.yml">
<img alt="GitHub Actions: CI" src=https://img.shields.io/github/actions/workflow/status/ray-chew/pyCSA/ci.yml?logo=github&label=CI&branch=main>
</a>
<a href="https://ray-chew.github.io/pyCSA/index.html">
<img alt="Documentation" src=https://img.shields.io/badge/docs-online-blue?logo=github>
</a>
<a href="https://www.gnu.org/licenses/gpl-3.0">
<img alt="License: GPL v3" src=https://img.shields.io/badge/License-GPLv3-blue.svg>
</a>
<a href="https://github.com/psf/black">
<img alt="Code style: black" src=https://img.shields.io/badge/code%20style-black-000000.svg>
</a>
<a href="https://doi.org/10.5281/zenodo.10877090">
<img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.10877090.svg">
</a>
</p>


The Constrained Spectral Approximation (CSA) method is a physically sound and robust method for approximating the spectrum of subgrid-scale orography. It operates under the following constraints:

* Utilises a limited number of spectral modes (no more than 100)
* Significantly reduces the complexity of physical terrain by over 500 times
* Maintains the integrity of physical information to a large extent
* Compatible with unstructured geodesic grids
* Inherently scale-aware

This method is primarily used to represent terrain for weather forecasting purposes, but it also shows promise for broader data analysis applications.

---

**[Read the documentation here](https://ray-chew.github.io/pyCSA/index.html)**

---

## Requirements

See [`requirements.txt`](https://github.com/ray-chew/pyCSA/blob/main/requirements.txt)

> **NOTE:**  The Sphinx dependencies can be found in [`docs/source/conf.py`](https://github.com/ray-chew/pyCSA/blob/main/docs/source/conf.py).


## Usage

### Installation

Install the latest release from PyPI:

```bash
pip install pycsa-specappx
```

The distribution is named `pycsa-specappx` (the bare `pycsa` name was already taken on PyPI by an unrelated project), but the import name is unchanged — `import pycsa`.

To run the bundled experiment scripts in [`runs/`](https://github.com/ray-chew/pyCSA/tree/main/runs) / [`examples/`](https://github.com/ray-chew/pyCSA/tree/main/examples), or to contribute, work from a clone instead:

```bash
git clone https://github.com/ray-chew/pyCSA && cd pyCSA
pip install -e ".[test]"
```

### Configuration

Run parameters are assembled programmatically inside the run scripts using the [`pycsa.config.params`](https://github.com/ray-chew/pyCSA/blob/main/pycsa/config/params.py) dataclass. Example experiment scripts live in [`runs/`](https://github.com/ray-chew/pyCSA/tree/main/runs) and [`examples/`](https://github.com/ray-chew/pyCSA/tree/main/examples); the reusable building blocks are in the [`pycsa`](https://github.com/ray-chew/pyCSA/tree/main/pycsa) package (`pycsa.core`, `pycsa.wrappers`, `pycsa.plotting`, `pycsa.data`, `pycsa.compute`).

Runs that read on-disk data (e.g. the global ICON+ETOPO pipeline) locate it through `SPEC_APPX_*` environment variables, which are read by `pycsa/local_paths.py` (copied from `local_paths.py.template`):

```bash
export SPEC_APPX_DATA_DIR=/path/to/data          # directory containing the ICON grid
export SPEC_APPX_ETOPO_DIR=/path/to/data/etopo_15s
export SPEC_APPX_MERIT_DIR=/path/to/MERIT        # MERIT runs only
export SPEC_APPX_REMA_DIR=/path/to/REMA          # MERIT runs only
export SPEC_APPX_OUTPUT_DIR=/path/to/outputs
```

Set these directly or with `source setup_paths.sh`. The bundled [`examples/`](https://github.com/ray-chew/pyCSA/tree/main/examples) need no such setup — their data ships with the repo.

### Execution

A simple setup can be found in [`runs/idealised_isosceles.py`](https://github.com/ray-chew/pyCSA/blob/main/runs/idealised_isosceles.py), a fixed-seed idealised benchmark. From a clone, run it directly:

```console
python -m runs.idealised_isosceles
python3 ./runs/idealised_isosceles.py
```

However, the codebase is structured such that the user can easily assemble a run script to define their own experiments. Refer to the documentation for the [available APIs](https://ray-chew.github.io/pyCSA/api.html).

### Examples

Three self-contained examples ship with bundled data (no download needed):

- **Idealised** — `pycsa-idealised` (synthetic terrain with a known spectrum; see the [tutorial](https://ray-chew.github.io/pyCSA/tutorial.html)).
- **Aleutians / MERIT** — [`examples/icon_regional_minimal.py`](https://github.com/ray-chew/pyCSA/blob/main/examples/icon_regional_minimal.py) (real ICON cell, ~10 s).
- **Andes / ETOPO** — [`examples/icon_etopo_andes.py`](https://github.com/ray-chew/pyCSA/blob/main/examples/icon_etopo_andes.py) (real ETOPO cell with ocean-aware masking; see the [showcase](https://ray-chew.github.io/pyCSA/showcase.html)).

## License

GNU GPL v3 (tentative)

## Contributions

Refer to the open issues that require attention.

Any changes, improvements, or bug fixes can be submitted to upstream via a pull request.

