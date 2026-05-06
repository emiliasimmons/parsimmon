# parsimmon

**Par**ameter and **sim**ulation **mon**agement -- quickly define parameter and manage parameter sets with convenient results cache management.

![Persimmon by DianaWolfskin on Pixabay](docs/persimmon_small.png)

<details>
  <summary>LLM-guided setup for your project</summary>

  Copy this to your coding agent:

  ```
  Fetch and follow the instructions at https://github.com/InstituteforDiseaseModeling/parsimmon/blob/main/docs/llm_install.md
  ```
</details>


## Parameter Management

Parsimmon organizes simulations into four levels:

```
Manager      -- owns a simulation function and a registry of studies
  Study      -- a set of base parameters + named branches
    Branch   -- parameter overrides (may include ranges for sweeps)
      Trial  -- one fully-resolved parameter dict, ready to run
```

A **Manager** binds your simulation function to one or more **Studies**. Each Study holds base defaults and zero or more **Branches** -- named groups of parameter overrides. When a branch contains ranges (e.g. `psm.arange`, `psm.linspace`), parsimmon expands them via Cartesian product into individual **Trials**.

The default study is the last one registered, or whichever is marked `default=True`.

### Example driver file

```python
# run_sim.py
import parsimmon as psm
from model import run, default_pars       # your sim function + defaults

pm = psm.Manager(run, cache=True)

# base() returns your model's full parameter dict, expanded with overrides.
# default_pars() returns the model defaults; keyword arguments override them.
def base():
    return default_pars(
        sim={"n_agents": 5_000, "rand_seed": psm.arange(5)},
        flags={"verbose": False},
    )

@pm.study(extends=base)
def dose_response(st):
    dose = psm.linspace(0.1, 1.0, 5)
    st.branch("Control",   {"treatment": False})
    st.branch("Treatment", {"treatment": True, "dose": dose})
    return st

@pm.study(extends=base, default=True)
def adherence(st):
    uptake = psm.linspace(0.1, 1.0, 10)
    enroll = psm.each([0.02, 0.05, 0.10, 0.20])
    st.branch("Baseline", {"treatment": False})
    st.branch("Treated",  {"treatment": True, "adherence": {"uptake": uptake, "enroll_prob": enroll}})
    return st

@adherence.analysis
def plot_results(results):
    for branch_name, group in results.branches.items():
        print(f"{branch_name}: {len(group)} runs")

if __name__ == "__main__":
    pm.cli_run(jobs=4)
```

### CLI usage

```bash
python run_sim.py                         # run the default study
python run_sim.py dose_response           # run by name
python run_sim.py -j 4                    # 4 parallel workers
python run_sim.py -f                      # force re-run (ignore cache)
python run_sim.py -a adherence.uptake=0.5 # override a parameter (dot notation for nested)
python run_sim.py -l                      # list all registered studies
python run_sim.py -l [name]               # list details for one study
python run_sim.py --print                 # print all studies (fully expanded)
python run_sim.py --print [name]          # print one study
python run_sim.py -s                      # skip analysis after run
python run_sim.py --clean                 # remove stale cache entries
python run_sim.py --clean [name]          # remove stale entries for one study
python run_sim.py --certify               # certify stale cache entries
python run_sim.py --certify [name]        # certify one study
```

### In-process usage (Jupyter / console)

```python
from run_sim import pm, adherence

results = pm.run(adherence)               # run and get Results
results = pm.run(adherence, jobs=4)       # parallel
results = pm.run("adherence")             # by name
results = pm.run(adherence, force=True)   # ignore cache
results = pm.results(adherence)           # load from cache without running

pm.list()                                 # print study tree
pm.print_study(adherence)                 # print one study (fully expanded)
```


## Caching

Enable caching by passing `cache=True` to `Manager`. Results are stored under `data/cache/` by default. The default is `False`

### How it works

- **Cache key**: function of the canonical parameter dict (order-independent, supports nested dicts and numpy arrays).
- **fn_hash**: computed from the AST of your simulation function and all project-local modules it imports. Stored alongside each result. If the function changes, parsimmon detects staleness and prompts you to rerun.
- **Deduplication**: two studies that resolve to identical parameters share a single cached result.

### What gets hashed

The `fn_hash` boundary determines what invalidates the cache:

```
  HASHED (invalidates cache)          NOT HASHED (safe to change)
 +---------------------------------+ +-------------------------------+
 |  model.py                       | |  run_sim.py (driver)          |
 |    def run(pars, meta): ...     | |    pm = Manager(run, ...)     |
 |                                 | |    @pm.study                  |
 |  helpers.py (imported by model) | |    def experiment(st): ...    |
 |    def calc(...): ...           | |                               |
 |                                 | |  analysis functions           |
 |  (any module model.py imports   | |  plotting code                |
 |   from within your project)     | |  parameter definitions        |
 +---------------------------------+ +-------------------------------+
```

Keep your simulation function and its local imports in separate files from your driver/analysis code. Changes to the driver, parameter definitions, or analysis functions will not invalidate cached results.

The tree below shows which files are hashed. Everything `model.py` imports (recursively, within your project) is included. Driver files import `model.py` but are not themselves hashed.

```
# invalidates cache (tree shows imports -- all are hashed)
model.py
├── analyzers.py
│   └── utils.py
└── interventions.py

# safe to change
model_validation.py
├── model.py (imported, but model.py itself is unchanged)
└── plotlib.py
sensitivity_analysis.py
├── model.py
└── plotlib.py
calibration_zimbabwe.py
├── model.py
└── plotlib.py
```

### Custom serialization

The default serializer is `dill`. Override with custom `save`/`load` callables:

```python
import sciris as sc

pm = Manager(run, cache=SimFileCache("data/cache", save=sc.save, load=sc.load))
```

Both callables are used for result files and the index. Signatures: `save(path, obj) -> None`, `load(path) -> Any`.

### Custom cache backends

Subclass `SimCacheBase` for non-filesystem storage (SQLite, S3, etc.):

```python
from parsimmon import SimCacheBase

class MyBackend(SimCacheBase):
    def save(self, cache_key, result, metadata): ...
    def load(self, cache_key): ...
    def exists(self, cache_key): ...
    def index(self): ...
    def add_index_entry(self, metadata): ...
    def keys(self): ...
    def delete(self, cache_key): ...
    def clear(self): ...

pm = Manager(run, cache=MyBackend(...))
```


## API Reference

### Manager

| Method / Property | Description |
|---|---|
| `Manager(fn, path=None, cache=None, data_dir=None, plots_dir=None)` | Bind a simulation function. `cache=True` for default file cache. |
| `@pm.study(extends=None, default=False)` | Register a study builder. Returns a decorated function with `.analysis` sub-decorator. |
| `pm.run(study, jobs=None, force=False, do_analysis=True, overrides=None)` | Run a study (by name or function). Returns `Results`. |
| `pm.cli_run(jobs=None, argv=None)` | CLI entry point. Parses `sys.argv`. |
| `pm.results(study=None)` | Load cached `Results` without running. |
| `pm.list(study=None)` | Print study tree with branch/job counts. |
| `pm.print_study(study=None)` | Print study (fully expanded parameters). |
| `pm.certify(study=None)` | Update stored `fn_hash` to match current code. |
| `pm.clean(study=None, deep=False)` | Remove stale (or all) cache entries. |
| `pm.add_argument(*args, **kwargs)` | Add custom `argparse` arguments to `cli_run`. |
| `pm.data_dir` | Resolved data directory path. |
| `pm.plots_dir` | Resolved plots directory path. |

### Study

| Method / Property | Description |
|---|---|
| `Study(base=None)` | Create a study, optionally seeded with a parameter dict. |
| `st.defaults(overrides)` | Deep-merge into base parameters. |
| `st.branch(name, overrides=None)` | Register a named branch with parameter overrides. |
| `st.clear(name_or_keys, keys=None)` | Remove keys from a branch or the base. |
| `st.print_study()` | Pretty-print all branches with ranges resolved. |
| `st.branches` | List of registered branch names. |
| `len(st)` | Total trial count across all branches. |
| `iter(st)` | Yields `Trial` objects. |

### Trial

`Trial` is a `NamedTuple` yielded when iterating a Study:

| Field | Type | Description |
|---|---|---|
| `branch` | `str` | Branch name. |
| `sim_id` | `int` | 1-based index across all branches. |
| `branch_id` | `int` | 1-based index within the branch. |
| `pars` | `objdict` | Fully-resolved parameter dict. |

### Range helpers

All available as `psm.<fn>` and `Study.<fn>`.

| Function | Description |
|---|---|
| `arange(*args, ndigits=3)` | Step range (mirrors `np.arange`). |
| `linspace(*args, ndigits=3)` | Evenly-spaced values (mirrors `np.linspace`). |
| `logspace(*args, ndigits=3)` | Log-spaced values (mirrors `np.logspace`). |
| `each(iterable)` / `iter(iterable)` | Wrap any sequence as a sweep axis. |
| `link(source, fn)` | Derived parameter. `source` can be a range, key string, another link, or a list. |

Ranges trigger Cartesian expansion when a Study is iterated. Pass `ndigits=None` to skip rounding.

### Results

| Constructor | Description |
|---|---|
| `Results(path)` | Load from a cache directory. |
| `Results(cache_backend)` | Load from a `SimCacheBase` instance. |
| `pm.run(study)` | Returns `Results` after execution. |
| `pm.results(study)` | Returns `Results` from cache. |

| Method / Property | Description |
|---|---|
| `iter(results)` | Yields result values. Triggers loading. |
| `results.items()` | Yields `((pars, meta), value)` tuples. Triggers loading. |
| `results.first()` | Returns `((pars, meta), value)` for the first entry. Triggers loading. |
| `results.iter_params()` | Yields `(pars, meta)` without loading values. |
| `results[start:stop]` | Slice returns a new `Results` (no loading). |
| `len(results)` | Count without loading. |
| `results.filter(predicate)` | Filter by `P` expression or `lambda pars, meta: ...`. |
| `results.groupby(*keys)` | Group by parameter keys or `P` expressions. |
| `results.P` | Query builder -- tab-completable, supports `==`, `!=`, `<`, `>`, `<=`, `>=`, `&`, `\|`, `.unique()`. |
| `results.branches` | `objdict` mapping branch name to `Results`. |
| `results.studies` | `objdict` mapping study name to `Results`. |
| `results.<name>` | Attribute access filters by study or branch name. |

### Cache

| Class / Method | Description |
|---|---|
| `SimFileCache(directory, save=None, load=None)` | Built-in filesystem cache. Custom `save`/`load` override serialization. |
| `SimCacheBase` | Abstract base class for custom backends. |
| `.certify_entries(...)` | Mark entries as certified with current `fn_hash`. |
| `.clean_entries(...)` | Delete stale entries. |
| `.remove_orphans()` | Delete result files with no index entry. |
