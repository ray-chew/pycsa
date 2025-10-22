# Debug Scripts

This directory contains debugging and development scripts used during ETOPO/MERIT data loader development.

These are **not** automated tests - they are manual debugging scripts.

## Files

- `debug_etopo_load_cg.py` - Debug script for ETOPO coarse-grid data loading
- `compare_merit_etopo.py` - Comparison script between MERIT and ETOPO datasets

## Usage

These scripts are typically run directly for debugging purposes:

```bash
python tests/debug/debug_etopo_load.py
```

They are not included in the pytest test suite.
