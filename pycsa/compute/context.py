"""ComputeContext: explicit container for per-task compute resources.

Bundles the ``BufferPool`` (per-task lifetime; owned by the context) and an
accessor for the worker-local tile cache (worker-process lifetime; the
context only holds a getter). Threading a single ctx through the pipeline
makes the data flow explicit, replacing the previous pattern of implicit
``BufferPool`` creation inside ``get_pmf.__init__`` + module-level
``tile_cache.get_worker_cache()`` calls scattered through ``do_cell``.

The tile_cache field is a *callable* rather than the cache itself because
the real cache is a per-Dask-worker singleton owned by the worker process
and can't be carried across pickle boundaries. A test can substitute a
stub by passing any callable that returns a cache-like object.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pycsa.core.buffer_pool import BufferPool


@dataclass
class ComputeContext:
    """Per-task compute resources for the CSA pipeline.

    Parameters
    ----------
    buffer_pool
        Owned by the context. One ``BufferPool`` per ``ComputeContext``.
    tile_cache
        Optional callable that returns the worker-local tile cache. The
        cache itself is not stored (it's owned by the Dask worker process);
        we hold a getter that resolves at call time. Pass ``None`` for
        environments that don't need tile_cache (idealised runs, tests).
    prior
        Optional :class:`pycsa.core.priors.Prior` carried through to
        ``lin_reg.do``. Spike scripts thread a structured prior here
        without modifying any call site. ``None`` (default) means the
        preserved scalar-trace branch in ``lin_reg.do`` runs unchanged.
    selector
        Optional :class:`pycsa.core.mode_selection.ModeSelector`
        carried through to ``interface.second_appx``. Spike scripts
        thread an alternative selector here without modifying any
        call site. ``None`` (default) means the preserved inline
        ``argmax`` loop in ``second_appx`` runs unchanged.
    """

    buffer_pool: BufferPool = field(default_factory=BufferPool)
    tile_cache: Optional[Callable[[], Any]] = None
    prior: Optional[Any] = None
    selector: Optional[Any] = None

    @classmethod
    def default(cls) -> "ComputeContext":
        """Production setup: fresh ``BufferPool`` + tile_cache accessor.

        Imports ``pycsa.core.tile_cache`` lazily so callers in
        dependency-light environments can still construct a no-cache
        context via plain ``ComputeContext()``.
        """
        try:
            from pycsa.core.tile_cache import get_worker_cache

            return cls(buffer_pool=BufferPool(), tile_cache=get_worker_cache)
        except ImportError:
            return cls(buffer_pool=BufferPool(), tile_cache=None)


def _resolve_ctx(
    ctx: Optional[ComputeContext], buffer_pool: Optional[BufferPool]
) -> ComputeContext:
    """Backward-compat shim for callers still passing ``buffer_pool=``.

    Internal helper used by ``f_trans`` / ``lin_reg`` to accept either the
    new ``ctx=`` kwarg or the legacy ``buffer_pool=`` kwarg during the
    deprecation window. Emits a ``DeprecationWarning`` if the legacy form
    is used. Default-constructs a context when neither is supplied.
    """
    if ctx is not None:
        return ctx
    if buffer_pool is not None:
        warnings.warn(
            "buffer_pool= is deprecated; pass ctx=ComputeContext(buffer_pool=...) "
            "instead. The legacy kwarg will be removed in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )
        return ComputeContext(buffer_pool=buffer_pool)
    return ComputeContext()
