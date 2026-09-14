"""Pin every numeric library to one thread BEFORE numpy is imported.

Edge devices have one to four slow cores and share them with acquisition and
communication. Multi-threaded BLAS on a laptop would flatter every latency
figure and add scheduling jitter, so all benchmarks run single-threaded.
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
