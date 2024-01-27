
pushd pkg
rm -rf dist build wvpy.egg-info wvpy/__pycache__ tests/__pycache__
pip uninstall -y wvpy
pytest --cov wvpy > ../coverage.txt
cat ../coverage.txt
python3 setup.py sdist bdist_wheel
pip install --no-deps -e "$(pwd)"  # sym link to source files
pdoc -o docs ./wvpy
popd
twine check pkg/dist/*

