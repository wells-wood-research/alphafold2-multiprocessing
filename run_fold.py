import os
import os.path
import re
import warnings
import subprocess

warnings.filterwarnings("ignore")

from absl import logging

logging.set_verbosity("error")
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import pickle
import py3Dmol
import matplotlib.pyplot as plt
from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.data.tools import hhsearch
import ipywidgets
from ipywidgets import interact, fixed
import shutil
import json

from itertools import combinations, repeat

from multiprocessing import Pool


def run_model(query_sequence: str, num_models: int, jobname: str):
    """Predicts structure using AlphaFold2"""

    # remove whitespaces
    print(f"Starting {jobname}")
    query_sequence = "".join(query_sequence.split())
    query_sequence = re.sub(r"[^a-zA-Z]", "", query_sequence).upper()

    # remove whitespaces
    jobname = "".join(jobname.split())
    jobname = re.sub(r"\W+", "", jobname)

    with open(f"{jobname}.fasta", "w") as text_file:
        text_file.write(">1\n%s" % query_sequence.upper())

    # number of models to use

    msa_mode = "MMseqs2"  # @param ["MMseqs2","single_sequence","custom"]
    use_msa = True if msa_mode == "MMseqs2" else False
    use_custom_msa = True if msa_mode == "custom" else False

    # not installed
    use_amber = False  # @param {type:"boolean"}
    use_templates = False  # @param {type:"boolean"}
    # @markdown ---

    with open(f"{jobname}.log", "w") as text_file:
        text_file.write("num_models=%s\n" % num_models)
        text_file.write("use_amber=%s\n" % use_amber)
        text_file.write("use_msa=%s\n" % use_msa)
        text_file.write("msa_mode=%s\n" % msa_mode)
        text_file.write("use_templates=%s\n" % use_templates)

    if use_custom_msa and not os.path.isfile(f"{jobname}.custom.a3m"):
        custom_msa_dict = files.upload()
        custom_msa = list(custom_msa_dict.keys())[0]
        os.rename(custom_msa, f"{jobname}.custom.a3m")
        print(f"moving {custom_msa} to {jobname}.custom.a3m")

    # get msa
    os.system(f"./msa2.bsh {jobname}")

    # the following code is written by Sergey Ovchinnikov
    # setup the model
    """Not installed
  if use_amber and "relax" not in dir():
    sys.path.insert(0, '/usr/local/lib/python3.7/site-packages/')
    from alphafold.relax import relax"""

    if "model_params" not in dir():
        model_params = {}
    for model_name in ["model_1", "model_2", "model_3", "model_4", "model_5"][
        :num_models
    ]:
        if model_name not in model_params:
            model_config = config.model_config(model_name)
            model_config.data.eval.num_ensemble = 1
            model_params[model_name] = data.get_model_haiku_params(
                model_name=model_name, data_dir="."
            )
            if model_name == "model_1":
                global model_runner_1
                model_runner_1 = model.RunModel(model_config, model_params[model_name])
            if model_name == "model_3":
                global model_runner_3
                model_runner_3 = model.RunModel(model_config, model_params[model_name])

    if use_templates:
        template_features = mk_template(jobname)
    else:
        template_features = mk_mock_template(query_sequence)

    # parse MSA
    if use_msa:
        a3m_lines = "".join(open(f"{jobname}.a3m", "r").readlines())
    elif use_custom_msa:
        a3m_lines = "".join(open(f"{jobname}.custom.a3m", "r").readlines())
    else:

        shutil.copy2(f"{jobname}.fasta", f"{jobname}.a3m")
        a3m_lines = "".join(open(f"{jobname}.a3m", "r").readlines())

    msa, deletion_matrix = pipeline.parsers.parse_a3m(a3m_lines)

    # gather features
    feature_dict = {
        **pipeline.make_sequence_features(
            sequence=query_sequence, description="none", num_res=len(query_sequence)
        ),
        **pipeline.make_msa_features(msas=[msa], deletion_matrices=[deletion_matrix]),
        **template_features,
    }
    plddts = predict_structure(
        jobname, feature_dict, model_params=model_params, do_relax=use_amber
    )


def mk_mock_template(query_sequence):
    # since alphafold's model requires a template input
    # we create a blank example w/ zero input, confidence -1
    ln = len(query_sequence)
    output_templates_sequence = "-" * ln
    output_confidence_scores = np.full(ln, -1)
    templates_all_atom_positions = np.zeros(
        (ln, templates.residue_constants.atom_type_num, 3)
    )
    templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
    templates_aatype = templates.residue_constants.sequence_to_onehot(
        output_templates_sequence, templates.residue_constants.HHBLITS_AA_TO_ID
    )
    template_features = {
        "template_all_atom_positions": templates_all_atom_positions[None],
        "template_all_atom_masks": templates_all_atom_masks[None],
        "template_sequence": [f"none".encode()],
        "template_aatype": np.array(templates_aatype)[None],
        "template_confidence_scores": output_confidence_scores[None],
        "template_domain_names": [f"none".encode()],
        "template_release_date": [f"none".encode()],
    }
    return template_features


def mk_template(jobname):
    template_featurizer = templates.TemplateHitFeaturizer(
        mmcif_dir="templates/",
        max_template_date="2100-01-01",
        max_hits=20,
        kalign_binary_path="kalign",
        release_dates_path=None,
        obsolete_pdbs_path=None,
    )

    hhsearch_pdb70_runner = hhsearch.HHSearch(
        binary_path="hhsearch", databases=[jobname]
    )

    a3m_lines = "\n".join(open(f"{jobname}.a3m", "r").readlines())
    hhsearch_result = hhsearch_pdb70_runner.query(a3m_lines)
    hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)

    # print(hhsearch_hits)

    templates_result = template_featurizer.get_templates(
        query_sequence=query_sequence,
        query_pdb_code=None,
        query_release_date=None,
        hhr_hits=hhsearch_hits,
    )
    return templates_result.features


def set_bfactor(pdb_filename, bfac):
    I = open(pdb_filename, "r").readlines()
    O = open(pdb_filename, "w")
    for line in I:
        if line[0:6] == "ATOM  ":
            seq_id = int(line[23:26].strip()) - 1
            O.write(
                "{prefix}{bfac:6.2f}{suffix}".format(
                    prefix=line[:60], bfac=bfac[seq_id], suffix=line[66:]
                )
            )
    O.close()


def predict_structure(prefix, feature_dict, model_params, do_relax=True, random_seed=0):
    """Predicts structure using AlphaFold for the given sequence."""

    # Run the models.
    plddts = []
    unrelaxed_pdb_lines = []
    relaxed_pdb_lines = []

    for model_name, params in model_params.items():
        print(f"running {model_name}")
        # swap params to avoid recompiling
        # note: models 1,2 have diff number of params compared to models 3,4,5
        if any(str(m) in model_name for m in [1, 2]):
            model_runner = model_runner_1
        if any(str(m) in model_name for m in [3, 4, 5]):
            model_runner = model_runner_3
        model_runner.params = params

        processed_feature_dict = model_runner.process_features(
            feature_dict, random_seed=random_seed
        )
        prediction_result = model_runner.predict(processed_feature_dict)
        unrelaxed_protein = protein.from_prediction(
            processed_feature_dict, prediction_result
        )
        unrelaxed_pdb_lines.append(protein.to_pdb(unrelaxed_protein))
        plddts.append(prediction_result["plddt"])

        if do_relax:
            # Relax the prediction.
            amber_relaxer = relax.AmberRelaxation(
                max_iterations=0,
                tolerance=2.39,
                stiffness=10.0,
                exclude_residues=[],
                max_outer_iterations=20,
            )
            relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
            relaxed_pdb_lines.append(relaxed_pdb_str)

    # rerank models based on predicted lddt
    lddt_rank = np.mean(plddts, -1).argsort()[::-1]
    plddts_ranked = {}
    for n, r in enumerate(lddt_rank):
        print(f"model_{n+1} {np.mean(plddts[r])}")

        unrelaxed_pdb_path = f"{prefix}_unrelaxed_model_{n+1}.pdb"
        with open(unrelaxed_pdb_path, "w") as f:
            f.write(unrelaxed_pdb_lines[r])
        set_bfactor(unrelaxed_pdb_path, plddts[r] / 100)

        if do_relax:
            relaxed_pdb_path = f"{prefix}_relaxed_model_{n+1}.pdb"
            with open(relaxed_pdb_path, "w") as f:
                f.write(relaxed_pdb_lines[r])
            set_bfactor(relaxed_pdb_path, plddts[r] / 100)

        plddts_ranked[f"model_{n+1}"] = plddts[r]

    return plddts_ranked


if __name__ == "__main__":
    WORKERS = 34
    with open("/scratch/sequence-recovery-benchmark/monomers_af.json") as file:
        predicted_sequence_dict = json.load(file)

    with Pool(processes=WORKERS) as p:

        p.starmap(
            run_model,
            zip(
                list(predicted_sequence_dict.values()),
                repeat(1),
                list(predicted_sequence_dict.keys()),
            ),
        )
        p.close()
