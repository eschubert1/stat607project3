all: setup simulate analyze figures

simulate: src/simulation.py src/config.py
	python -c "from src.simulation import simulate_data; from src.config import ALL_SCENARIOS, RNG_STREAMS; simulate_data(ALL_SCENARIOS, RNG_STREAMS)"

analyze:
	python -c "from src.simulation import evaluate; from src.config import ALL_SCENARIOS; evaluate(ALL_SCENARIOS)"
	python -c "from src.simulation import save_intervals; from src.config import ALL_SCENARIOS; save_intervals(ALL_SCENARIOS)"

figures:
	python -c "from src.simulation import zipper_plot; zipper_plot()"
	python -c "from src.simulation import make_tables; make_tables()"

setup: requirements.txt
	pip install -r requirements.txt

clean:
	rm data/simulated/*.pkl
	rm results/raw/estimates/*.pkl
	rm results/raw/metrics/*.pkl
	rm results/figures/*.pdf

test:
	pytest tests/run_tests.py
	pytest tests/regression_test.py

profile:
	rm profiling/*.log
	kernprof -l -o profiling/baseline_profile.lprof profiling/baseline_profile.py
	python -m line_profiler profiling/baseline_profile.lprof > profiling/baseline_profiling_results.txt
	kernprof -l -o profiling/optimized_profile.lprof profiling/optimized_profile.py
	python -m line_profiler profiling/optimized_profile.lprof > profiling/optimized_profiling_results.txt

complexity:
	python complexity.py

benchmark:
	python benchmark.py

stability-check:
	rm -f warnings.log
	python -c "from src.simulation import simulate_data; from src.config import ALL_SCENARIOS, RNG_STREAMS; simulate_data(ALL_SCENARIOS, RNG_STREAMS)" "log"