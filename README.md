# CAFL4DS-E

Continuous, Active Federated Learning for Data Streams: a curated research snapshot
based on [aidotse/cafl4ds](https://github.com/aidotse/cafl4ds).

## Selected notebooks

- [00_ssl_filter_session_findings](notebooks/selected/00_ssl_filter_session_findings.ipynb)
- [02_health_regression_history](notebooks/selected/02_health_regression_history.ipynb)
- [09_health_prediction_regressions](notebooks/selected/09_health_prediction_regressions.ipynb)
- [68_single_image_filter_factorial](notebooks/selected/68_single_image_filter_factorial.ipynb)
- [91_single_image_selected_metrics](notebooks/selected/91_single_image_selected_metrics.ipynb)

The notebooks contain saved figures and can be viewed on GitHub without downloading
the experiment data. References to older studies inside the notebooks are retained
as historical context; those unselected notebooks are not part of this snapshot.

## Setup

Requires Python 3.10 or newer and [uv](https://docs.astral.sh/uv/).
From this repository's root:

```powershell
uv sync --locked --extra cpu --group dev --group analysis
uv run --no-sync python -m scripts.run_loop data=synthetic img_size=16 ssl=simsiam
uv run --no-sync pytest tests -o "addopts=-ra"
uv run --no-sync jupyter lab notebooks/selected
```

For the NVIDIA environment, select `--extra cu124` instead of `--extra cpu`.
These extras are mutually exclusive. See [development notes](docs/developing.md)
for other hardware configurations.

## What is included

- The Python package and experiment configurations in `cafl4ds/`.
- Unit tests and the runner/helper scripts referenced by the package, tests,
  configurations and selected studies.
- Selected notebooks, dependency lockfile, license, and referenced experiment protocols.

Datasets, downloaded video/images, model checkpoints, trained predictors,
experiment outputs, caches, logs, virtual environments, old report builders and
unselected notebooks are excluded. Research runners and report builders that read
`outputs/` require local artifacts; this repository does not bundle those artifacts.
Some historical protocols contain machine-specific paths that need local configuration.

The upload is a fresh source snapshot. Upstream Git history, old release/website
automation and generated documentation are not included. The source working copy
and its local experiment evidence remain separate from this publication checkout.

## License and attribution

Original project by Mauricio Munoz / AI Sweden. Distributed under the
[MIT License](LICENSE.md); the original copyright notice is preserved.
