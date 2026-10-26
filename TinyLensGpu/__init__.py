__version__ = "0.1.0"

# pyright: reportMissingImports=false

import os
# Force JAX backend for caskade
os.environ['CASKADE_BACKEND'] = 'jax'
