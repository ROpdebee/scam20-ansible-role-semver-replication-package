# Ansible SemVer replication package

Replication package accompanying the SCAM 2020 publication: R. Opdebeeck, A. Zeraouli, C. Velázquez-Rodríguez, C. De Roover. “Does Infrastructure as Code Adhere to Semantic Versioning? An Analysis of Ansible Role Evolution”, In Proc. 20th Int. Working Conf. on Source Code Analysis and Manipulation, 2020.

## Directory structure

- `ansible_semver` contains the pipeline used to gather, extract, and analyse Ansible roles in the study.
- `notebooks` contains the notebook and related code used to analyse the extracted data.
- `random-forest` contains code, data, and output of the Random Forest classifier.

## Dataset
The dataset used in the paper, containing a large collection of Ansible Galaxy roles, their source code, versions, and metadata, can be found at [https://doi.org/10.5281/zenodo.4039514](https://doi.org/10.5281/zenodo.4039514). This dataset also contains numerous files that can be used in intermediary phases of the pipeline.
