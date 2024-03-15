build_cython:
	rm -rf build && python setup.py build_ext && python setup.py bdist_wheel
build:
	poetry build
# cython_infer:
# 	rm -rf build && python setup_infer_only.py build_ext && python setup.py bdist_wheel
clean:
	rm -rf build dist

pull_build_install:
	git pull origin dev && make build && pip install --no-deps --force-reinstall dist/*.whl