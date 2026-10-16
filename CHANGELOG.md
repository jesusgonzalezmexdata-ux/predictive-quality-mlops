# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-10-16

### Added
- MIT license, Model Card and English README with the Spanish version kept as `README.es.md`.
- Project page on GitHub Pages and a live demo on Hugging Face Spaces, published automatically from `main`.
- Release workflow: notes from this changelog and an evidence bundle attached to every tag.

### Removed
- C-MAPSS module, data and reports: the work lives in predictive-maintenance-survival-cmapss.

## [1.0.0] - 2026-10-05

### Added
- SECOM lot-risk model with chronological validation, cost-based threshold and release / monitor / hold taxonomy.
- Holdout evaluated once: PR-AUC 0.089 against a 0.054 failure rate; the saving is not statistically demonstrated and is reported as such.
- Drift monitoring, FastAPI service, Streamlit dashboard, prediction log, Docker and CI.
