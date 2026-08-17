# CX3/PBS reference scripts

These files preserve the final training and evaluation arguments used for the
dissertation. They are templates, not universally portable launchers.

Before submission:

1. Set `ROOT` to the repository checkout.
2. Set `VENV_PATH` or replace the environment activation line.
3. Adapt queue, GPU type, walltime, and log directives to the local scheduler.
4. Download the checkpoint from Zenodo and use its verified path.
5. Export `HF_TOKEN` only in the job environment; never write it into a script.

The original server paths and notification email were removed from this public
copy. Sampler profiles, seeds, budgets, and benchmark stopping rules were not
changed.

