.PHONY: cpp-build cpp-test cpp-bench py-install py-test lint verify bench bench-update clean

cpp-build:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j

cpp-test: cpp-build
	ctest --test-dir build --output-on-failure

cpp-bench: cpp-build
	./build/tinyinfer_bench

py-install:
	pip install -e ".[dev]"

py-test:
	pytest -v

lint:
	ruff check python tests benchmarks

verify: cpp-test lint py-test

bench:
	python3 benchmarks/bench_kernels.py

bench-update:
	python3 benchmarks/bench_kernels.py --update-readme README.md

clean:
	rm -rf build .pytest_cache .ruff_cache *.egg-info
