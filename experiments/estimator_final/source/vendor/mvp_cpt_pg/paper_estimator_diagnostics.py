"""Entry point for current fixed-budget diagnostics.

Use scripts/verify_revised_design.py or run_paper_experiments.py --experiment 2.
The former random-level diagnostics are preserved under
output_paper/experiment_audit_20260914/source_snapshot and are not the current method.
"""
def run_diagnostics(*args, **kwargs):
    raise RuntimeError('Use python scripts/verify_revised_design.py --output NEW_DIRECTORY')
