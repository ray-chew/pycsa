#!/bin/bash
# Setup script for local paths
# Usage: source setup_paths.sh

# Detect if we're on HPC or local machine
if [[ -n "$SLURM_JOB_ID" ]] || [[ -n "$PBS_JOBID" ]] || [[ $(hostname) == *"hpc"* ]]; then
    echo "Detected HPC environment"
    export SPEC_APPX_ENV="HPC"

    # HPC paths - UPDATE THESE FOR YOUR HPC
    export SPEC_APPX_DATA_DIR="${HOME}/pyCSA/data"
    export SPEC_APPX_OUTPUT_DIR="${HOME}/pyCSA/outputs"
    export SPEC_APPX_MERIT_DIR="${HOME}/pyCSA/data/MERIT"
    export SPEC_APPX_REMA_DIR="${HOME}/pyCSA/data/REMA"
    export SPEC_APPX_ETOPO_DIR="${HOME}/pyCSA/data/etopo_15s/"
else
    echo "Detected local environment"
    export SPEC_APPX_ENV="LOCAL"

    # Local paths - UPDATE THESE FOR YOUR LOCAL MACHINE
    export SPEC_APPX_DATA_DIR="${HOME}/pyCSA/data"
    export SPEC_APPX_OUTPUT_DIR="${HOME}/pyCSA/outputs"
    export SPEC_APPX_MERIT_DIR="${HOME}/pyCSA/data/MERIT"
    export SPEC_APPX_REMA_DIR="${HOME}/pyCSA/data/REMA"
    export SPEC_APPX_ETOPO_DIR="${HOME}/pyCSA/data/etopo_15s/"
fi

echo "Environment: $SPEC_APPX_ENV"
echo "Data directory: $SPEC_APPX_DATA_DIR"
echo "Output directory: $SPEC_APPX_OUTPUT_DIR"

# Create local_paths.py if it doesn't exist
if [ ! -f "pycsa/local_paths.py" ]; then
    echo "Creating pycsa/local_paths.py from template..."
    cp pycsa/local_paths.py.template pycsa/local_paths.py
fi
