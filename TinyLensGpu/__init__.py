__version__ = "0.1.0"

# pyright: reportMissingImports=false

import os
# Force JAX backend for caskade
os.environ['CASKADE_BACKEND'] = 'jax'

# JAX arrays intentionally do not define truthiness, but a few legacy helpers
# still use ``array or default`` when selecting optional kernels. Keep the
# compatibility shim narrow so explicit ``None`` checks remain preferred in new
# code while old call sites keep working.
try:
    import jax
    import numpy as np
    from jaxlib._jax import ArrayImpl

    def _tinylens_array_bool(self):
        if self.shape == ():
            return bool(np.asarray(self))
        return True

    jax.Array.__bool__ = _tinylens_array_bool
    ArrayImpl.__bool__ = _tinylens_array_bool
except Exception:
    pass
