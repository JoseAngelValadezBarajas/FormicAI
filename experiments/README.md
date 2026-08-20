# Experiment Records

This directory is for small, versioned reproducibility records for future FormicAI experiments.

Track:

- experiment protocol and selection criteria
- dataset/source/model SHA256 hashes
- source roles and split policy
- checkpoint-selection metadata
- environment metadata from `formicai.utils.environment`
- dependency lock or `pip freeze` snapshot for the actual experiment environment
- final result summaries

Do not track:

- model weights
- raw video
- extracted images
- YOLO label/image datasets
- large generated analysis artifacts

Future experiment freezes should capture environment metadata, capture and hash an environment lock such as `pip freeze`, and store those small records with the experiment manifest. Historical v1-v4 environments are not retroactively reconstructed here.
