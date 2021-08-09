
pushd pkg
rm -rf dist build wvpy.egg-info wvpy/__pycache__ tests/__pycache__
pip uninstall -y wvpy
pytest --cov wvpy > ../coverage.txt
cat ../coverage.txt
python3 setup.py sdist bdist_wheel
pip install dist/wvpy-*.tar.gz
pdoc -o docs wvpy
popd
twine check pkg/dist/*

