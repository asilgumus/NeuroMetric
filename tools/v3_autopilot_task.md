# Authorized BrainAGE research and improvement continuation

The user explicitly requested this unattended workflow after verified early
stopping, or completion at the epoch limit with best-checkpoint validation
within-one-year performance <=70%. Read the supplied evidence and current files.
Do not claim 70% is achievable, and do not change labels, evaluation tolerances
or split definitions to manufacture success.

USER PRIORITY UPDATE: maximize rounded-year exact matches, then +/-1-year
coverage, with MAE tracked as a tradeoff. This is an explicitly authorized
prospective selection change, not grounds for rewriting old run records.
v3_train.py now additionally saves best_hits.pt and hit_selection.json with
lexicographic (rounded_year_match, within_1_year, -mae) selection including
epoch-zero parent. Preserve legacy MAE best.pt and terminal verification.
Before the next submission compare completed checkpoints on the same fixed
validation participants, document rounded hits and +/-1 counts, and consider
the best exact-hit parent, not automatically the lowest-MAE parent.
Research a justified train-only loss/decoder modification targeting this
objective; document before submitting one sequential experiment. Preserve
chronological labels, np.rint semantics and continuous +/-1 definition.
No Alzheimer/OASIS, IXI test or DLBS results may guide optimization. Do not
restart the currently running contrast-cont4 job just to change its objective;
its existing checkpointing is MAE-only and cannot recover unsaved epochs.

1. Research primary academic sources on improving SFCN brain-age prediction.
   Record citations, findings and a bounded experiment rationale under artifacts/v3.
   Analyze validation predictions by age and dataset; never open IXI test or DLBS
   for tuning. Current train data: IXI/SALD/NIMH, approximately 923 usable people.
2. Inspect v3_train.py, v3_model.py and builder before modifying them. Existing
   refine run unfreezes conv_4/conv_5/head with rates 3e-6/1e-5/1e-4, uses shifts,
   mirroring, dropout, frozen BatchNorm, KL+.05 MAE and plateau scheduler.
   Preserve all user edits and every prior checkpoint. Use apply_patch for edits.
3. Implement the most defensible next experiment, with unit tests and a documented
   selection protocol. Use at most one new Kaggle GPU job initially; no paid
   services, new cloud resources, public data release or unlimited parallel search.
   New job must have a unique slug, QC verification and checksum-pinned parent.
   Keep existing best checkpoint if no improvement. Keep fixed validation metrics
   including MAE, +/-1 year, +/-5 years and rounded-year match. Changes in any one
   metric are not equivalent to general improvement; explain tradeoffs.
4. Run .venv/bin/python -m unittest discover -s tests and an appropriate smoke
   check before submitting. If correct and ready, submit the private Kaggle job.
   Start persistent monitoring of that exact job at 3, then 8, then 10 minute
   intervals using tools/v3_training_watch.py, with a separate output tag.
   Observation failure is not job failure; never restart a live job.
5. If the successor is block-only SFCN, arrange tools/v3_autopilot.py for its exact
   ref to check every 15 minutes and reapply the user's terminal conditions.
   Otherwise implement and test equally strict stop-reason verification before
   enabling the successor watcher. Avoid duplicate watchers or agent launches.
6. Save concrete results, submitted ref, service names and limitations in the
   final summary. If blocked by auth/quota/missing authority, record the blocker;
   do not silently pretend that research or training succeeded.

Do not edit the dashboard or unrelated systems. Read applicable AGENTS/skills.
No subagents unless separately authorized. Respect the risk of repeated
validation tuning: keep a documented experiment ledger and bound each experiment.
The user explicitly requires continuation after subsequent early stopping or
epoch-limit completion with within-one-year <=70%. Do NOT invent a global
one-successor budget or launch successor watchers with --terminal-only.
One sequential GPU experiment per handoff is allowed, not unlimited parallel
search. Re-research and choose a justified next experiment at each handoff;
do not merely keep repeating identical training. When meaningful progress is
truly blocked, record the evidence instead of silently disabling continuation.
Do not promise 70% attainability or clinical validity.
