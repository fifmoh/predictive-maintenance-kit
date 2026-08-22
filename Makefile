.PHONY: help setup check test smoke run docs demo clean clean-room data leakage-check reproduce evaluate report ui-test colab-bundle
PY := uv run --no-sync --python 3.11

help:
	@printf "%s\n" "setup check test smoke run docs demo clean clean-room data leakage-check reproduce evaluate report ui-test colab-bundle"
setup:
	uv sync --all-extras
	@test -n "$(RESEARCHOPS_WHEEL)" -a -n "$(RESEARCHOPS_SHA256)"
	@printf "%s  %s\n" "$(RESEARCHOPS_SHA256)" "$(RESEARCHOPS_WHEEL)" | shasum -a 256 -c -
	uv pip install --reinstall "$(RESEARCHOPS_WHEEL)"
check:
	uv lock --check
	$(PY) ruff format --check src tests
	$(PY) ruff check src tests
	$(PY) mypy src
test:
	$(PY) pytest
smoke:
	$(PY) python -m predictive_maintenance.cli smoke
docs:
	$(PY) python -m predictive_maintenance.cli docs
demo:
	$(PY) python -m predictive_maintenance.cli demo
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist
clean-room:
	$(PY) python -m predictive_maintenance.cli smoke

data:
	$(PY) python -m predictive_maintenance.cli data
leakage-check:
	$(PY) pytest tests/test_protocol.py
reproduce:
	$(PY) python -m predictive_maintenance.cli reproduce
evaluate:
	$(PY) python -m predictive_maintenance.cli evaluate
report:
	latexmk -pdf -interaction=nonstopmode -output-directory=report report/technical-note.tex
ui-test:
	$(PY) pytest tests/test_protocol.py
colab-bundle:
	$(PY) python -m predictive_maintenance.cli colab-bundle
run:
	$(PY) streamlit run src/predictive_maintenance/app.py --server.address 127.0.0.1 --server.port 8504
