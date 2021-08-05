conda create --name fold python=3.7
conda deactivate
conda activate fold
chmod u+x dependency_install.bsh
chmod u+x msa2.bsh
bash dependency_install.bsh
conda install -c conda-forge jq
conda install -c conda-forge curl
conda install cudatoolkit cudnn cupti 
cd alphafold_/
pip install -r requirements.txt
pip install .
cd ../
pip install ipywidgets
pip install matplotlib
