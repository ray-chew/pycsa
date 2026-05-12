#!/usr/bin/env python3
"""
Check SLURM resource allocation for the current job.
"""

import os
import subprocess


def get_slurm_allocation():
    """Get SLURM resource allocation for current job."""

    # Check if running under SLURM
    job_id = os.environ.get("SLURM_JOB_ID")

    if not job_id:
        print("Not running in a SLURM job")
        return None

    print(f"SLURM Job ID: {job_id}")
    print("=" * 60)

    # Get info from environment variables
    info = {
        "Job ID": os.environ.get("SLURM_JOB_ID"),
        "Job Name": os.environ.get("SLURM_JOB_NAME"),
        "Partition": os.environ.get("SLURM_JOB_PARTITION"),
        "Nodes": os.environ.get("SLURM_JOB_NUM_NODES"),
        "CPUs per Task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "Total CPUs": os.environ.get("SLURM_NTASKS"),
        "Memory per Node (MB)": os.environ.get("SLURM_MEM_PER_NODE"),
        "Memory per CPU (MB)": os.environ.get("SLURM_MEM_PER_CPU"),
        "CPUs on Node": os.environ.get("SLURM_CPUS_ON_NODE"),
        "Tasks per Node": os.environ.get("SLURM_TASKS_PER_NODE"),
    }

    print("\nEnvironment Variables:")
    for key, value in info.items():
        if value:
            print(f"  {key:25s}: {value}")

    # Calculate total memory
    mem_per_node_mb = os.environ.get("SLURM_MEM_PER_NODE")
    num_nodes = os.environ.get("SLURM_JOB_NUM_NODES", "1")

    if mem_per_node_mb:
        mem_mb = int(mem_per_node_mb)
        mem_gb = mem_mb / 1024
        total_mem_gb = mem_gb * int(num_nodes)
        print(f"\n  Total Memory Allocated   : {total_mem_gb:.1f} GB ({mem_mb} MB)")

    # Get more details using scontrol
    try:
        result = subprocess.run(
            ["scontrol", "show", "job", job_id], capture_output=True, text=True
        )

        if result.returncode == 0:
            output = result.stdout

            # Parse key fields
            for line in output.split("\n"):
                if "MinMemoryNode=" in line:
                    # Extract memory
                    parts = line.split()
                    for part in parts:
                        if "MinMemoryNode=" in part:
                            mem_str = part.split("=")[1]
                            print(f"\n  MinMemoryNode (scontrol) : {mem_str}")

                if "NumCPUs=" in line:
                    parts = line.split()
                    for part in parts:
                        if part.startswith("NumCPUs="):
                            cpus = part.split("=")[1]
                            print(f"  NumCPUs (scontrol)       : {cpus}")

    except Exception as e:
        print(f"\nCouldn't get scontrol info: {e}")

    print("=" * 60)

    return info


if __name__ == "__main__":
    get_slurm_allocation()
