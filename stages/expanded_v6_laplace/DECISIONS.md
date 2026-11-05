# Laplace on Tier B: decisions (PROTOCOL.md is frozen; implementation choices and ambiguities are recorded here)
D1. Analytic Hessians are used for classification and regression (no laplace-torch call in the fit loop). They are checked against autograd by tierb_laplace.py checks
    (four checks, run at every start); the earlier study (expanded_v2) verified the same classification Hessian against laplace-torch 0.2.3 to 1e-7. Equivalent, not a deviation of method.
D2. Regression: predicted noise = softplus(raw variance output) as in Tier-B conversion; clamped at 1e-6 only inside the Hessian and the validation selection (avoids division by zero).
    Predictive variance = noise + phi^T cov phi on the standardised scale, then times scale^2 (as Tier B). The mean is unchanged.
D3. Tier-B Ctx objects load the test tensors into memory (inherited constructor). The fit stage never references c.test, c.raw_test or c.mask_test; the FROZEN marker records test_unopened_by_laplace.
D4. alpha grid 1e-4..1e4 is fixed by the protocol. If the selected alpha is at a grid edge it is reported (frozen/*.json 'alpha_at_grid_boundary'), not silently extended. Any extended grid
    would be a labelled sensitivity analysis, decided by the student, never a replacement of the primary result. Smoke test (bace, esol; scaffold; seed 0): both selected alpha = 1e-4, the lower edge.
D5. Novelty distance: Morgan radius 2, 2048 bits, no chirality, to the training molecules of the split in question (RDKit current version). Query molecules RDKit cannot parse (the 8 Tox21 SMILES of Tier-B D4)
    get d = 1 (maximally novel); unparsable training molecules are left out of the reference set. Novel subgroup = similarity < 0.4, i.e. d > 0.6.
D6. Novelty variants follow expanded_v2: 'Laplace constant' = Laplace variance + lambda; 'Novelty only' = lambda * d^2; 'Novelty Laplace' = Laplace variance + lambda * d^2. lambda grid [0, .01, .1, 1, 10, 100] per task,
    validation NLL, after alpha. A lambda at the grid's upper edge is recorded in frozen/NOV_*.json ('lambda_at_upper_edge'), not extended.
