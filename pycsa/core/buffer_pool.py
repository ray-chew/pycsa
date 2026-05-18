"""
Dynamic buffer pool for reusing NumPy arrays across multiple computations.

This module provides memory-efficient buffer management for spectral approximation
computations where array sizes may vary between cells (e.g., different amounts of
topography data per cell).
"""

import copy

import numpy as np


class BufferPool:
    """Dynamic buffer pool that auto-grows to handle variable array sizes.

    Strategy:
    - Keeps the largest buffer seen for each key
    - Returns views (slices) for smaller requests → zero-copy!
    - Auto-grows when larger size requested
    - Tracks usage statistics for performance analysis

    This is particularly effective for workflows processing many cells with
    varying data sizes, as it eliminates repeated memory allocations while
    adapting to size variations.

    Examples
    --------
    >>> pool = BufferPool()
    >>> # First call allocates
    >>> arr1 = pool.get_or_create('coeff', (1000, 100), np.float64)
    >>> # Second call with same size reuses buffer
    >>> arr2 = pool.get_or_create('coeff', (1000, 100), np.float64)
    >>> # Smaller size returns a view of existing buffer
    >>> arr3 = pool.get_or_create('coeff', (500, 100), np.float64)
    >>> # Larger size triggers reallocation
    >>> arr4 = pool.get_or_create('coeff', (2000, 100), np.float64)
    """

    def __init__(self):
        """Initialize empty buffer pool."""
        self.buffers = {}  # key -> (max_shape, array)
        self.stats = {}  # key -> {hits, misses, grows}

    def get_or_create(self, key, shape, dtype=np.float64):
        """Get buffer from pool, creating or growing as needed.

        Parameters
        ----------
        key : str
            Identifier for this buffer (e.g., 'coeff', 'E_tilda_lm')
        shape : tuple of int
            Requested shape for the array
        dtype : numpy dtype, optional
            Data type for the array (default: np.float64)

        Returns
        -------
        numpy.ndarray
            Array of requested shape and dtype. May be a view into a larger buffer.

        Notes
        -----
        The returned array should be treated as writable. If you need the data
        to persist beyond the next call to get_or_create with the same key,
        make a copy.
        """
        # Initialize stats for new keys
        if key not in self.stats:
            self.stats[key] = {"hits": 0, "misses": 0, "grows": 0}

        if key in self.buffers:
            current_shape, buf = self.buffers[key]

            # Check if requested size fits in current buffer
            if all(req <= curr for req, curr in zip(shape, current_shape)):
                # Cache hit! Return view of existing buffer
                self.stats[key]["hits"] += 1
                # Create view with appropriate slice for each dimension
                slices = tuple(slice(0, s) for s in shape)
                return buf[slices]

            # Need bigger buffer - reallocate
            self.stats[key]["grows"] += 1
            # Keep maximum of current and requested for each dimension
            new_shape = tuple(max(c, r) for c, r in zip(current_shape, shape))
            self.buffers[key] = (new_shape, np.empty(new_shape, dtype=dtype))

            # Return view of newly allocated buffer
            slices = tuple(slice(0, s) for s in shape)
            return self.buffers[key][1][slices]

        # First allocation for this key
        self.stats[key]["misses"] += 1
        self.buffers[key] = (shape, np.empty(shape, dtype=dtype))
        return self.buffers[key][1]

    def clear(self):
        """Free all buffers and reset statistics.

        Use this when done processing a batch of cells to release memory.
        In Dask workflows, buffers are automatically released when the
        worker process terminates, so calling clear() is optional.
        """
        self.buffers.clear()
        self.stats.clear()

    def get_stats(self):
        """Get buffer usage statistics for performance analysis.

        Returns
        -------
        dict
            Dictionary mapping buffer keys to statistics:
            - 'hits': Number of times buffer was reused
            - 'misses': Number of times buffer was allocated
            - 'grows': Number of times buffer was grown

        Examples
        --------
        >>> pool = BufferPool()
        >>> # ... use pool ...
        >>> stats = pool.get_stats()
        >>> print(f"Coefficient buffer hit rate: {stats['coeff']['hits'] /
        ...       (stats['coeff']['hits'] + stats['coeff']['misses']):.1%}")
        """
        return copy.deepcopy(self.stats)

    def get_memory_usage(self):
        """Get current memory usage of all buffers.

        Returns
        -------
        dict
            Dictionary with:
            - 'total_mb': Total memory used by all buffers in MB
            - 'buffers': Dict mapping keys to individual buffer sizes in MB
        """
        total_bytes = 0
        buffer_sizes = {}

        for key, (shape, buf) in self.buffers.items():
            size_bytes = buf.nbytes
            total_bytes += size_bytes
            buffer_sizes[key] = size_bytes / (1024**2)  # Convert to MB

        return {"total_mb": total_bytes / (1024**2), "buffers": buffer_sizes}
