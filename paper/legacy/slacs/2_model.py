#%%
import os
import pickle
import gzip
import sys
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["JAX_ENABLE_X64"] = "False"

# If you want to run on CPU, uncomment the following lines
# os.environ["NPROC"] = "1" #https://github.com/jax-ml/jax/issues/743
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["JAX_PLATFORMS"] = "cpu"

from TinyLensGpu.RunModel.RunLensModel import RunLensModel

with open("lens_names.txt", "r") as f:
    lens_names = f.readlines()
lens_names = [name.strip() for name in lens_names]

for ii, lens_name in enumerate(lens_names):
    if ii in range(0, 29): continue

    config_path = f'results/{lens_name}/modeling_config.yaml'
    lens_model = RunLensModel(config_path)

    print(f'============lens {ii}: {lens_name} begin running============')
    lens_model.run() 

    with gzip.open(f'results/{lens_name}/lens_model.pkl.gz', 'wb') as f:
        pickle.dump(lens_model, f)
    
    print(f'============lens {ii}: {lens_name} done============')
    sys.stdout.flush()

# %%
