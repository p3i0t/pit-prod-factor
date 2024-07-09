.PHONY: clean build build_cython build_install build_install_cython build_all

clean:
	rm -rf build/ dist/ *.egg-info/ 

build_cython:
	python setup.py bdist_wheel

build:
	poetry build

build_install: build
	pip install --no-deps --force-reinstall dist/*-py3-none-any.whl

build_install_cython: build_cython
	pip install --no-deps --force-reinstall dist/*cp*.whl

build_all: clean build_cython build