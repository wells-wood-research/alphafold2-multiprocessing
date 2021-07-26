conda create --name fold python=3.7
conda activate fold
mkdir AF2
cd AF2
chmod u+x dependency_install.bsh
./dependency_install.bsh
conda install -c conda-forge jq
conda install -c conda-forge curl
conda install cudatoolkit cudnn cupti 
cd ./alphafold_/
pip install .
cd ../
pip install ipywidgets
