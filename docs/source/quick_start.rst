Quickstart
==========
A quick and dirty guide to using the CSA codebase

Requirements
^^^^^^^^^^^^
To run the code, make sure the following packages are installed, preferably in a virtual environment.

.. literalinclude:: ../../requirements.txt

.. note::
    The Sphinx dependencies can be found in ``docs/source/conf.py``.

Overview
^^^^^^^^
The pyCSA codebase is structured modularly, see :numref:`structure` for a graphical overview.

The subpackage :mod:`pycsa.wrappers` provides interfaces to the core code components in :mod:`pycsa.core` and the plotting helpers in :mod:`pycsa.plotting`. For example, it defines the First and Second Approximation steps in the CSA algorithm and applies the tapering of the physical data. Refer to the :doc:`APIs <api>` for more details.

Helper functions and data structures are provided for the processing of user-defined topographies (:class:`pycsa.data.cell.topo`), grids (:class:`pycsa.data.cell.grid`), and input parameters (:class:`pycsa.config.params.params`).

These *building blocks* are then assembled for different kinds of experiments in user-defined run scripts. Some examples live in the top-level ``runs/`` and ``examples/`` directories.

.. graphviz::
    :align: center
    :name: structure
    :alt: pyCSA program structure
    :caption: pyCSA program structure

    digraph {
        graph [fontname="Verdana", fontsize="12"];
        node  [fontname="Verdana", fontsize="11", shape=box, style=rounded];
        edge  [fontname="Sans", fontsize="9"];

        data     [label="pycsa.data\n(grid / topo cells)",
                  URL="_autosummary/pycsa.data.cell.html", target="_top"];
        config   [label="pycsa.config\n(run parameters)",
                  URL="_autosummary/pycsa.config.params.html", target="_top"];
        core     [label="pycsa.core\n(delaunay, fourier,\nlin_reg, reconstruction)",
                  URL="_autosummary/pycsa.core.html", target="_top"];
        compute  [label="pycsa.compute\n(ComputeContext)",
                  URL="_autosummary/pycsa.compute.html", target="_top"];
        wrappers [label="pycsa.wrappers\n(interface.get_pmf,\ndiagnostics)",
                  URL="_autosummary/pycsa.wrappers.html", target="_top"];
        plotting [label="pycsa.plotting\n(plotter, cart_plot)",
                  URL="_autosummary/pycsa.plotting.html", target="_top"];
        runs     [label="runs / examples\n(experiment scripts,\npycsa-idealised CLI)"];

        subgraph cluster_inputs {
            margin=8;
            label = <<B>building blocks</B>>;
            data; config; core; compute;
        }

        data     -> wrappers;
        config   -> wrappers;
        core     -> wrappers;
        compute  -> wrappers;
        core     -> plotting [style=dashed];
        wrappers -> runs;
        plotting -> runs;
    }

A first run
^^^^^^^^^^^

The quickest way to see the method end-to-end is the bundled idealised
isosceles experiment. After ``pip install -e .`` it is a single command via the
``pycsa-idealised`` console script:

.. code-block:: console

    pycsa-idealised

The equivalent direct invocations also work:

.. code-block:: console

    python -m runs.idealised_isosceles
    python3 ./runs/idealised_isosceles.py

See the :doc:`tutorial` for a step-by-step walkthrough of this experiment, and
:doc:`hpc_reproducibility` for the global ICON+ETOPO pipeline.

.. note::

    Run parameters are assembled programmatically in the run scripts under
    ``runs/`` and ``examples/`` (using :class:`pycsa.config.params.params`),
    rather than via a separate input module. The design favours debugging and
    diagnostics in an interactive ``ipython`` environment.