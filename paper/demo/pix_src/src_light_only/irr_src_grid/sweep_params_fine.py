
import subprocess
import itertools
import statistics

cg_tols = [1e-5]
cg_maxiters = [300]
slq_probes_list = [16, 24, 32]
slq_steps_list = [25, 30, 35, 40]

results = []

for cg_tol, cg_maxiter, slq_probes, slq_steps in itertools.product(cg_tols, cg_maxiters, slq_probes_list, slq_steps_list):
    print(f"Testing: cg_tol={cg_tol}, cg_maxiter={cg_maxiter}, slq_probes={slq_probes}, slq_steps={slq_steps}")
    
    # Run 3 times to get average
    diffs = []
    for _ in range(3):
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
                    diffs.append(diff)
        except subprocess.CalledProcessError as e:
            print(f"  Error: {e}")
            
    if diffs:
        avg_diff = statistics.mean(diffs)
        std_diff = statistics.stdev(diffs) if len(diffs) > 1 else 0
        results.append({
            "cg_tol": cg_tol,
            "cg_maxiter": cg_maxiter,
            "slq_probes": slq_probes,
            "slq_steps": slq_steps,
            "avg_diff": avg_diff,
            "std_diff": std_diff
        })
        print(f"  Avg Difference: {avg_diff:.6f} (±{std_diff:.6f})")

# Sort by avg_diff
results.sort(key=lambda x: x['avg_diff'])

print("\nTop 5 configurations:")
for res in results[:5]:
    print(res)
