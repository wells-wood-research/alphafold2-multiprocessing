Before starting, please read the disclaimer at the end.

# Installing AF2 locally

## Dependencies and MSA
You can skip this section if you want to use our settings.

1. from https://colab.research.google.com/drive/1LVPSOf4L502F21RWBmYJJYYLDlOU2NTL?usp=sharing#scrollTo=a-COJivqdM8V copy the dependency cell into a file called "dependency_install.bsh"
2. Modify "dependency_install.bsh" with your settings. We use E_AMBER=False, USE_MSA=True, USE_TEMPLATES=False

## Start Installing
Simply run
```
bash start.sh
```

This assumes you have conda installed.

## Running Multiple Structures on the same GPU (Multiprocessing)

Running 1 structure at the time takes about 315MB of GPU. Using 
multiprocessing you could potentially run more structures on different 
workers.

```
python run_fold.py --workers 30 --num_models 1 --input_file /scratch/sequence-recovery-benchmark/monomers_af.json
```

`run_fold.py` accepts both .json or .fasta files

# Credits 

This work is hacked together by [Rokas Petrenas](https://github.com/rokaske199)and [Leonardo Castorina](https://github.com/universvm) from the ColabFold notebook from which `dependency_install.bsh`, `msa2.bsh` and `run_fold.py` are obtained. `run_fold.py` was modified to allow for multiprocessing and running multiple structures automatically.

As with ColabFold we would like to credit and thank:

- RoseTTAFold and AlphaFold team for doing an excellent job open sourcing the software. 
- Also credit to [David Koes](https://github.com/dkoes) for his awesome [py3Dmol](https://3dmol.csb.pitt.edu/) plugin, without whom these notebooks would be quite boring!
- A colab by Sergey Ovchinnikov (@sokrypton), Milot Mirdita (@milot_mirdita) and Martin Steinegger (@thesteinegger).

# Disclaimer

As per https://twitter.com/thesteinegger/status/1420055602970075138 be 
mindful of how you use this repository. The API is currently supported by 
only one server handling multiple thousands of requests per day. Refrain 
from using this tool until they have improved the API (we will keep this up 
to date!) 