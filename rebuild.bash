
pushd pkg
rm -rf dist build wvpy.egg-info wvpy/__pycache__ tests/__pycache__
pip uninstall -y wvpy
python3 setup.py sdist bdist_wheel
pip install dist/wvpy-*.tar.gz
popd
pdoc -o docs pkg/wvpy
pytest
pytest --cov pkg/wvpy pkg/tests > coverage.txt
cat coverage.txt
twine check pkg/dist/*

