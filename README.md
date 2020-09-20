# Dataset of Ansible roles

A dataset of Ansible roles accompanying the SCAM 2020 publication: R. Opdebeeck, A. Zeraouli, C. Velázquez-Rodríguez, C. De Roover. “Does Infrastructure as Code Adhere to Semantic Versioning? An Analysis of Ansible Role Evolution”, In Proc. 20th Int. Working Conf. on Source Code Analysis and Manipulation, 2020.

The full dataset can be found under the `all-ansible-roles` directory.
It contains the following files and directories:

- `repos`: The git repositories of all roles included in the dataset. The subdirectories are named using the role's Ansible Galaxy qualified name, i.e., `<namespace>.<role_name>`
- `roles.json`: A JSON file containing metadata extracted from Ansible Galaxy for each role.
- `repo_paths.json`: A mapping from Ansible Galaxy role IDs to their path in the `repos` directory.
- `tag_versions.json`: A mapping from Ansible Galaxy role IDs to the role's repository's git tags and metadata on these tags.
- `version_analysis.json`: Similar to `tag_versions.json`, but with additional filtering applied.
- `versiondiff_analysis.json`: Contains syntactical change statistics for each version bump in the role repositories.
- `structural_diff_analysis.json`: Contains structural change statistics for each version bump in the role repositories.
- `struct_diff_cache`: Directory containing per-role diff statistics, primarily used for caching during the pipeline.
- `metrics_diffs_releases.csv`: CSV containing the structural diff statistics merged with the bump type of the version increments.
- `reports`: Graphs and charts describing some of the output of the pipeline, as well as CSVs containing raw data.
- `version.json`: The version of the dataset structure.
