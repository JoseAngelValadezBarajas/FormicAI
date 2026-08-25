# V5 Training Source 001 Supply Audit

Status: PASS after protocol-hygiene remediation

Experiment: `ants_v5_crowdedscale_e01`
Source role: `TRAINING_SOURCE`

Important boundary: the V5 final test remains `FROZEN_UNOBSERVED`. No files under `datasets/external_test/v5_final_test/` were opened. The final-test source video was not opened, decoded, hashed, or used. Alias comparison used only the already recorded final-test source SHA from `experiments/ants_v5/final_test_freeze.md`.

## Protocol Incident

Classification: `PROTOCOL_HYGIENE_DEVIATION` and `NO_SCIENTIFIC_CONTENT_LEAKAGE`.

During the initial training-source audit, an overly broad source inventory surfaced filesystem metadata for the forbidden V5 final-test source once:

- filename: `videonuevoants.mp4`
- file size: surfaced in terminal output only
- filesystem modified time: surfaced in terminal output only

The final-test source video was not opened, decoded, hashed, probed, summarized, or inferred on. No final-test images, labels, predictions, metrics, visual statistics, annotation counts, or detector outputs were read or exposed, and `datasets/external_test/v5_final_test/` was not read.

The surfaced metadata did not influence candidate extraction, source acceptance, strata design, source-role assignment, selection policy, frame count, temporal bins, candidate-pool generation, manual visual characterization, or any future pruning recommendation. The candidate extraction used only `samples/v2/ants_v5_training_source_001.mp4`.

Remediation added a reusable source-role guard that blocks final-test/forbidden sources before filesystem metadata inspection. For training-source audit/design, the permitted alias-safety mechanism is comparison against already-recorded registry identifiers and SHA values, not rereading the forbidden source.

## Source

- filename: `ants_v5_training_source_001.mp4`
- path: `samples/v2/ants_v5_training_source_001.mp4`
- SHA256: `53a659c615fb1ae5e6ab1ae96ba84531cb8e7d8aae6c396a19b8f7e7cf623e7b`
- size: 153755857 bytes
- resolution: 1920x1080
- FPS: 59.975746657223736
- frame count: 3633
- stream duration: 60.574511 seconds
- container duration: 60.690729 seconds
- codec: h264
- recording metadata: motorola edge 60 pro / Android 16 / creation_time 2026-08-25T05:08:56Z

## Alias Check

- exact alias status: PASS, no exact SHA match found
- known historical hash records compared: 44
- unique known hashes compared: 19
- matches: 0
- includes recorded hashes for Ants2-Ants11, ant2/ant4/own-colony sources, public-source registries where available, and the recorded v5 final-test source SHA only

## Role Guard

The source-role registry now represents this video as:

- role: `TRAINING_SOURCE`
- train_eligible: true
- development_validation_eligible: false
- checkpoint_selection_eligible: false
- observed_reference: false
- final_test: false

Observed references and final test remain outside train eligibility.

The registry also records an explicit `forbiddenForTrainingSourceAudit` guard for `v5_final_test` / `videonuevoants.mp4`. The guard blocks opening, decoding, hashing, image extraction, stat-based metadata collection, video probing, and content-derived inspection before filesystem access.

## Candidate Pool

- candidate count: 120
- decoded frames: 120
- unique image SHA256: 120
- staging directory: `artifacts/analysis/ants_v5_training_source_001/candidate_frames/`
- contact sheets: `artifacts/analysis/ants_v5_training_source_001/contact_sheets/`

Extraction policy:

Divide the full video frame interval into 120 equal temporal bins and extract the center frame from each bin. This uses only timing/frame count, not model predictions, ant counts, perceived difficulty, or final-test information.

Temporal coverage:

- first candidate: frame 15, t=0.250s
- last candidate: frame 3617, t=60.306s
- approximate spacing: 0.505s

## Image-Only / Manual Supply Characterization

Exact density, bbox scale, and local-neighbor categories require manual annotations. No detector output was used, and the candidate records are marked `requires_manual_annotation_for_exact_density_scale_crowding`.

Descriptive visual review of the contact sheets:

- sparse: present but not dominant; exact count requires labels
- medium: visually common across the full timeline
- crowded: visually present, especially around the left debris/tube cluster and around interactions near the larger ant
- small-heavy: visually present through multiple small workers and edge/tube ants; exact bbox-area class requires labels
- mixed: strongly present; large/medium ant plus smaller workers appear in the same scene
- medium/large-heavy: present because the large ant remains visible through much of the sequence
- isolated-dominant: some individual moving ants are visible away from clusters
- moderate local crowding: visually common
- crowded-neighbor: visually present in the left cluster and near-ant interactions; exact neighbor count requires labels

Domain coverage:

- realistic own-domain plastic/tube setting: yes
- strong glare/reflection: yes
- warm/yellow cast: yes
- partial occlusion/tube edge: yes
- small ants with nearby neighbors: visually likely
- sparse/medium balance: present, but final balance depends on later pruning and annotation

## Supply Sufficiency

Likely sufficient for the preregistered V5 objective: true, with a caveat.

The video appears to provide the missing real-camera crowded/small-worker exposure better than the sparse v4 addition. However, it is one visually stable scene, so final frame selection must aggressively prune redundancy. If manual annotation shows the candidate pool cannot produce enough distinct crowded/small-neighbor frames, an additional never-holdout training recording should be captured.

Need additional training recording right now: uncertain.

## Redundancy / Near-Duplicate Findings

- adjacent dHash near-duplicate pairs at Hamming <= 4: 104
- interpretation: high temporal redundancy, expected for a single stable 60-second scene

Future pruning policy:

1. Keep source-level role guard first; never mix with validation/reference/final-test sources.
2. Start from the 120-frame candidate pool, not the full video.
3. Require a minimum temporal spacing such as >= 2 seconds unless a frame has a materially different manually observed scene arrangement.
4. Cluster by perceptual hash/visual similarity and cap near-identical arrangements.
5. Preserve real variation in glare, focus, ant-density, local crowding, occlusion, and camera/tube position.
6. After manual annotation, enforce density/scale/crowding balance using the preregistered strata.
7. Do not use detector predictions at any pruning step.

## Boundary

No detector inference, pseudo-labeling, automatic annotation, Roboflow upload, final subset selection, or training was performed.

V5 final test accessed: false.
