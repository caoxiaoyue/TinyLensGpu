#%%
import os
import pickle
import gzip
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# If you want to run on CPU, uncomment the following lines
# os.environ["NPROC"] = "1" #https://github.com/jax-ml/jax/issues/743
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["JAX_PLATFORMS"] = "cpu"

from TinyLensGpu.RunModel.RunLensModel import RunLensModel
config_path = 'model_config.yaml'

lens_model = RunLensModel(config_path)
lens_model.run() 

with gzip.open('output/lens_model.pkl.gz', 'wb') as f:
    pickle.dump(lens_model, f)

# %%
