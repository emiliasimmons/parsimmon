# parsimmon -- LLM-Guided Setup

You are helping a user integrate **parsimmon** into their simulation project.
Follow these steps in order. Do not skip steps.

For full API documentation, see the `README.md` in the parsimmon repository.

---

## Step 1: Install

Verify the user is in a virtual environment before installing. If they are on
system Python, warn them and get explicit confirmation before proceeding.

Install parsimmon:

```bash
pip install git+https://github.com/InstituteforDiseaseModeling/parsimmon.git
```

Confirm:

```bash
python -c "from parsimmon import Manager; print('ok')"
```

If the import fails, resolve the error before continuing.

---

## Step 2: Git safety

Run `git status` in the project root. If the working tree is dirty, stop and
help the user commit or stash before making any changes. If the directory is
not a git repository, warn the user that changes cannot be easily reverted and
offer to initialize one.

---

## Step 3: Understand the separation

Parsimmon requires a clean separation between two sides:

### Hashed side (invalidates cache when changed)

The **simulation function** and everything it imports from within the project.
This is the function passed to `Manager(fn)`. Its signature:

```python
def fn(pars: dict, metadata: dict) -> result:
    ...
```

`pars` is the fully resolved parameter dictionary for one trial.
`metadata` contains `study`, `branch`, `sim_id`, and `branch_id`.

Parsimmon computes an `fn_hash` from the AST of this function and all
project-local modules it imports, recursively. If any of that code changes,
cached results are detected as stale. This is the hashed boundary.

### Unhashed side (safe to change freely)

Everything else: driver files, parameter definitions, study configuration,
analysis functions, plotting code. Changes to these do not invalidate cached
results.

### Default parameters

A function that returns the full default parameter dictionary. This can live:

- **In the model file** (recommended) -- changes to defaults invalidate the
  cache, which is usually what you want.
- **In the driver file** -- defaults don't affect cache validity.
- **In a dedicated parameters file** -- same as driver unless the model
  imports it.

The placement depends on whether the user wants default changes to trigger
re-runs. Ask them.

### Driver files

A driver file creates a `Manager`, defines studies, and runs them. A project
can have multiple driver files (e.g. `calibration.py`, `sensitivity.py`,
`validation.py`) that all import the same simulation function. The canonical
pattern:

```python
import parsimmon as psm
from model import run, default_pars

pm = psm.Manager(run, cache=True)

def base():
    return default_pars(
        sim={"n_agents": 5_000, "rand_seed": psm.arange(5)},
        flags={"verbose": False},
    )

@pm.study(extends=base)
def experiment(st):
    st.branch("Control",   {"treatment": False})
    st.branch("Treatment", {"treatment": True, "dose": psm.linspace(0.1, 1.0, 5)})
    return st

@experiment.analysis
def plot_results(results):
    for branch_name, group in results.branches.items():
        print(f"{branch_name}: {len(group)} runs")

if __name__ == "__main__":
    pm.cli_run(jobs=4)
```

### Parameter types

When caching is enabled, every value in the parameter dictionary must be
**reproducible** -- parsimmon hashes parameters to compute cache keys. The
following types are supported:

- `dict` (any nesting depth, order-independent)
- `list`, `tuple`
- `np.ndarray` (dtype + shape + bytes)
- `int`, `float`, `bool`, `str`, `None`, `bytes`
- numpy scalars (`np.int64`, `np.float64`, etc. -- coerced to Python types)

The following are **rejected with a `TypeError`**:

- Callables (functions, lambdas, `functools.partial`, classes with `__call__`)
- Objects whose `repr()` contains a memory address (`0x...`)
- Objects that cannot be deepcopied (file handles, locks, generators)
- Objects whose repr is not stable across copies (RNG state, mutable singletons)

If the user's parameters contain any of these, they need to be refactored:
replace callables with names/flags that resolve to the callable inside the
simulation function; replace RNG objects with integer seeds.

### Caching

Recommend `cache=True`. It stores results under `data/cache/` by default,
deduplicates across studies with identical parameters, and detects staleness
via `fn_hash`. The user can always run with `-f` to force re-runs or disable
caching entirely. See the README for custom serialization and cache backends.

---

## Step 4: Explore and interview

Before making any changes, explore the user's project and interview them
relentlessly about every aspect of the integration until you reach a shared
understanding. Walk through each branch of the refactoring decisions, resolving
dependencies below one at a time. For each question, explain **why** it matters
and provide your recommended answer based on what you've seen in the code. If
a question can be answered by reading the project, read the project instead of
asking.

Do not proceed to implementation until all of these are resolved:

1. **What is the simulation function?** Which function or method is the core
   simulation logic? What does it accept and return? Does it need to be wrapped
   to match the `fn(pars, metadata)` signature? If the project has multiple
   simulation functions, recommend consolidating them into a single
   parameter-driven function where a flag or parameter selects the behavior.
   This is the parsimmon model: one simulation function, many parameter
   configurations. That said, different driver files *can* use different
   simulation functions with separate `Manager` instances, so if consolidation
   doesn't make sense, work with the user to decide which function each driver
   should use.

2. **What are the parameters?** Where do defaults live now -- inline constants,
   config dicts, dataclasses, config files? Do any parameter values contain
   non-reproducible types (callables, RNG objects, objects with identity-based
   repr)? These must be refactored before caching will work.

3. **What is the import graph?** What does the simulation function import from
   within the project? All of those modules are inside the hashed boundary --
   changes to any of them invalidate the cache. Are there modules currently
   imported by the simulation function that should not be (analysis code,
   plotting utilities, parameter definitions that should live outside the
   boundary)?

4. **Where does analysis and plotting live?** These must stay outside the
   hashed boundary. Are there analysis functions tangled into the simulation
   module that need to be extracted?

5. **Where should default parameters live?** In the model file (recommended --
   changes invalidate cache), in the driver file, or in a separate module?
   Does the user want default changes to trigger re-runs?

6. **How many driver files?** Does the project have one entry point or several
   (calibration, sensitivity analysis, validation, etc.)? Each can be a
   separate driver file sharing the same simulation function.

7. **Layout preferences.** Adapt to the user's existing directory structure. Do
   not impose a new layout unless their project has no clear structure or they
   ask for one.

---

## Step 5: Implement

Execute the agreed-upon plan:

1. Work file by file. Do not scatter unrelated edits across a large diff.
2. After all changes, run any existing tests (`python -m pytest` or the
   project's test command). Fix failures before proceeding.
3. Run `python <driver_file> --list` for each driver file to verify studies
   are registered and expand correctly.
4. Do a single trial run to confirm the simulation function works with the
   parsimmon signature.
5. Show the user the output and confirm the integration is working.
