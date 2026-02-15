
import subprocess
import itertools

cg_tols = [1e-4, 1e-5]
cg_maxiters = [200, 300]
slq_probes_list = [16, 32, 64]
slq_steps_list = [40, 60, 80]

results = []

for cg_tol, cg_maxiter, slq_probes, slq_steps in itertools.product(cg_tols, cg_maxiters, slq_probes_list, slq_steps_list):
    print(f"Testing: cg_tol={cg_tol}, cg_maxiter={cg_maxiter}, slq_probes={slq_probes}, slq_steps={slq_steps}")
    
    cmd = [
        "python", "tune_params.py",
        "--cg_tol", str(cg_tol),
        "--cg_maxiter", str(cg_maxiter),
        "--slq_probes", str(slq_probes),
        "--slq_steps", str(slq_steps)
    ]
    
    try:
        output = subprocess.check_output(cmd, text=True)
        for line in output.splitlines():
            if "Difference:" in line:
                diff = float(line.split(":")[1].strip())
                results.append({
                    "cg_tol": cg_tol,
                    "cg_maxiter": cg_maxiter,
                    "slq_probes": slq_probes,
                    "slq_steps": slq_steps,
                    "diff": diff
                })
                print(f"  Difference: {diff}")
    except subprocess.CalledProcessError as e:
        print(f"  Error: {e}")

# Sort by difference
results.sort(key=lambda x: x['diff'])

print("\nTop 5 configurations:")
for res in results[:5]:
    print(res)
