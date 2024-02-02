cython:
	rm -rf build && python setup.py build_ext && python setup.py bdist_wheel
# cython_infer:
# 	rm -rf build && python setup_infer_only.py build_ext && python setup.py bdist_wheel
clean:
	rm -rf build dist