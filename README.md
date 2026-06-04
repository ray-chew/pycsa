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

Fork this repository and clone your remote fork, then `pip install -e .`.

### Configuration

Run parameters are assembled programmatically inside the run scripts using the [`pycsa.config.params`](https://github.com/ray-chew/pyCSA/blob/main/pycsa/config/params.py) dataclass. Example experiment scripts live in [`runs/`](https://github.com/ray-chew/pyCSA/tree/main/runs) and [`examples/`](https://github.com/ray-chew/pyCSA/tree/main/examples); the reusable building blocks are in the [`pycsa`](https://github.com/ray-chew/pyCSA/tree/main/pycsa) package (`pycsa.core`, `pycsa.wrappers`, `pycsa.plotting`, `pycsa.data`, `pycsa.compute`).

### Execution

A simple setup can be found in [`runs/idealised_isosceles.py`](https://github.com/ray-chew/pyCSA/blob/main/runs/idealised_isosceles.py). After `pip install -e .` the easiest way to run it is via the console script:

```console
pycsa-idealised
```

The equivalent direct invocations also work:

```console
python -m runs.idealised_isosceles
python3 ./runs/idealised_isosceles.py
```

However, the codebase is structured such that the user can easily assemble a run script to define their own experiments. Refer to the documentation for the [available APIs](https://ray-chew.github.io/pyCSA/api.html).

## License

GNU GPL v3 (tentative)

## Contributions

Refer to the open issues that require attention.

Any changes, improvements, or bug fixes can be submitted to upstream via a pull request.

