

conda remove --name wvpy_dev_env --all --yes
conda env create -f wvpy_dev_env.yaml
conda activate wvpy_dev_env
pip install --no-deps -e "$(pwd)/pkg"  # sym link to source files


