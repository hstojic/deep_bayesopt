# Deep Bayesian optimisation

## Overview

This repository accompanies paper "High throughput Bayesian optimisation with uncertainty aware deep neural networks".


### Running experiments

We rely on Hydra for configuring the experiments and Ray to parallelise the jobs.
Here are the commands to execute the experiments:


## Cite

Check in again soon :)


## Installation


This project uses [Poetry](https://python-poetry.org/docs) to
manage dependencies in a local virtual environment. To install Poetry, [follow the
instructions in the Poetry documentation](https://python-poetry.org/docs/#installation).

To install this project in editable mode, run the commands below from the root directory repository.

```bash
poetry install
```

Poetry `install` command creates a virtual environment for this project
in a hidden `.venv` directory under the root directory.

You must also run the `poetry install` command to install updated dependencies when
the `pyproject.toml` file is updated, for example after a `git pull`.


### Adding new Python dependencies

- To specify dependencies required by `deep_bayesopt`, run `poetry add`.  
- To specify dependencies required to build or test the project, run `poetry add --dev`.


## Development notes

We work on feature branches and merge to `main` branch directly. Pull requests need to be submitted and approved to merge in the branch. Quality checks stated below need to pass before approving the request.

Quality checks are as follows. We write and run tests with [pytest](https://pytest.org) and use [type hints](https://docs.python.org/3/library/typing.html) for documentation and static type checking with [mypy](http://mypy-lang.org). We do this throughout the source code and tests. We format all Python code, other than the notebooks, with [black](https://black.readthedocs.io/en/stable/), [flake8](https://flake8.pycqa.org/en/latest/), and [isort](https://pycqa.github.io/isort/).

Here are the commands for essential development tasks. Run them from the root directory of this repository (after following the installation steps). 

Run reformatting of the code using black, flake8 and isort with
```bash
poetry run task format
```

Run the type checker with
```bash
poetry run task mypy
```

To run the full test suite, including Mypy, run: 

```bash
poetry run task test
```

Alternatively, you can run just the unit tests, starting with the failing tests and exiting after
the first test failure:

```bash
poetry run task quicktest
```

[GitHub actions](https://docs.github.com/en/actions) will automatically run all the quality checks against pull requests to the main branch. The GitHub repository is set up such that these need to pass in order to merge.


# License

[Apache License 2.0](LICENSE)
