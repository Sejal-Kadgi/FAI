"""Run the complete experimental pipeline from scratch:

    python run_all.py

1. clears results/cache (cached equilibria and RNR solutions are recomputed),
2. runs E1, E2, E3 (testbed S), E4, E5, E3 (testbed L) in that order,
3. regenerates every figure (figures/) and table / number file (results/),
4. writes results/run_info.json (versions, timings).

Every experiment uses fixed random seeds, so a rerun reproduces the same
numbers. Wall-clock time of the verification run on 2 CPU cores: 1.87 hours
(results/run_info.json).
"""
import json, os, platform, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(ROOT, "experiments")
RES = os.path.join(ROOT, "results")

STEPS = [
    ("E1", ["e1_cfr.py"], "e1.log"),
    ("E2", ["e2_ident.py"], "e2.log"),
    ("E3-S", ["e3_frontier.py", "small"], "e3_small.log"),
    ("E4", ["e45_learners.py", "e4"], "e4.log"),
    ("E5", ["e45_learners.py", "e5"], "e5.log"),
    ("E3-L", ["e3_frontier.py", "large"], "e3_large.log"),
]


def versions():
    import numpy, matplotlib
    return dict(python=platform.python_version(), numpy=numpy.__version__,
                matplotlib=matplotlib.__version__, platform=platform.platform(),
                cpus=os.cpu_count())


def main():
    os.makedirs(RES, exist_ok=True)
    cache = os.path.join(RES, "cache")
    if os.path.isdir(cache):
        shutil.rmtree(cache)
        print("cleared results/cache (everything is recomputed from scratch)")
    info = dict(versions=versions(), started=time.strftime("%Y-%m-%d %H:%M:%S"), steps={})
    t_all = time.time()
    for name, args, log in STEPS:
        t = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {name}: python {' '.join(args)}  (log: results/{log})", flush=True)
        with open(os.path.join(RES, log), "w", newline="\n") as fh:
            rc = subprocess.call([sys.executable] + args, cwd=EXP, stdout=fh, stderr=subprocess.STDOUT)
        info["steps"][name] = dict(seconds=round(time.time() - t, 1), returncode=rc)
        if rc != 0:
            print(f"{name} FAILED (exit {rc}); see results/{log}")
            sys.exit(rc)
        print(f"    done in {time.time() - t:.0f}s", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] figures, tables and report numbers: python plots.py", flush=True)
    rc = subprocess.call([sys.executable, os.path.join(ROOT, "plots.py")], cwd=ROOT)
    info["steps"]["plots"] = dict(returncode=rc)
    info["total_seconds"] = round(time.time() - t_all, 1)
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(RES, "run_info.json"), "w", newline="\n") as fh:
        json.dump(info, fh, indent=1)
    print(f"all done in {info['total_seconds']/3600:.2f} h")
    sys.exit(rc)


if __name__ == "__main__":
    main()
