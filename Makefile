.PHONY: cpp-build cpp-test cpp-bench cpp-cli py-install py-test lint verify bench bench-attn bench-e2e bench-update dashboard clean

cpp-build:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j

cpp-test: cpp-build
	ctest --test-dir build --output-on-failure

cpp-bench: cpp-build
	./build/tinyinfer_bench

cpp-cli: cpp-build
	./build/tinyinfer_cli --synthetic --dtype q8_0 --max-tokens 32

py-install:
	pip install --no-build-isolation -e ".[dev]" || pip install -e ".[dev]"

py-test:
	pytest -v

lint:
	ruff check python tests benchmarks dashboard

verify: cpp-test lint py-test

bench:
	python3 benchmarks/bench_kernels.py

bench-attn:
	python3 benchmarks/bench_attention.py

bench-e2e:
	python3 benchmarks/bench_e2e_compare.py --out-json benchmarks/results/latest.json

bench-update:
	python3 benchmarks/bench_kernels.py --update-readme README.md
	python3 benchmarks/bench_e2e_compare.py --out-json benchmarks/results/latest.json
	python3 benchmarks/update_readme_e2e.py --json benchmarks/results/latest.json --readme README.md

dashboard:
	python3 dashboard/server.py --port 8080

clean:
	rm -rf build .pytest_cache .ruff_cache *.egg-info
