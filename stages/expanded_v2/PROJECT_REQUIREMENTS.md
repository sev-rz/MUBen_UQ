# Standing project requirements

Recorded 4 October 2026 at the user's request.

- Treat this as a reproduction/implementation of MUBen first, with extensions separately identified.
- In every revised technical report, compare every reproduced dataset/model/method against the corresponding original paper table or pinned authors' CSV. Include original and reproduced mean/SD where available, absolute differences, metric conventions, exact source locations, and comparability caveats. Missing upstream SD is unavailable, not zero.
- Connect findings to specific claims of the original paper and state whether each is supported, contradicted, or not tested within the reproduced scope. Do not infer a general contradiction from a small subset.
- Maintain a fidelity matrix covering architecture, descriptors, data/splits, masks, training, checkpoint selection, uncertainty settings, seeds, metrics and dependencies. Resolve paper/source disagreements explicitly; keep paper-faithful and code-faithful variants separate.
- Prioritize greater fidelity and coverage, including regression, additional original UQ methods and a pretrained backbone when feasible. Record compute requirements and measured pilot timing before scaling.
- Preserve completed runs and their pre-test protocol. Existing BACE/BBBP results have been inspected: newly designed extensions on these tests are exploratory. Freeze new hypotheses/settings before opening any additional evaluation set; test labels must not tune methods.
- Report negative findings and uncertainty; do not promise an improvement or claim unpublished novelty without evidence.
- Keep protocol and execution evidence in the existing SC Hub project. Do not run heavy computation on its login node.

## Current authorized and completed scope

The user selected three datasets (BACE, BBBP, ESOL), five-member ensembles, CPU-only execution in the existing allocation, and the chemical-novelty extension. The completed expanded study is outputs/expanded_v2 locally and work/expanded_v2 in the Lustre SC Hub project. Read its README, PROTOCOL, FIDELITY and technical report before further changes. Preserve the earlier study separately.
