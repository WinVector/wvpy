
pushd pkg
rm -rf dist build wvpy.egg-info wvpy/__pycache__ tests/__pycache__
pip uninstall -y wvpy
python3 setup.py sdist bdist_wheel
pip install dist/wvpy-*.tar.gz
pdoc -o docs wvpy
pytest --cov tests > ../coverage.txt
cat coverage.txt
popd
twine check pkg/dist/*

