# Copy this file to local_config.py and edit. Task1/Task2 notebooks apply it after their own
# CONFIG cell, so notebook updates never overwrite your settings. Delete lines you don't need.

# --- dry run (small region, 1 member) ---
WORK_ROOT       = "./pyrosai_work_dryrun"
OUT_DIR         = "./pyrosai_output_dryrun"
MODELS          = ["MIROC-ES2H", "CESM2-WACCM"]
MAX_MEMBERS     = 1
BBOX            = (-125, -100, 30, 55)
RUN_ATTRIBUTION = False

# --- full run: comment out the block above and use this instead ---
# WORK_ROOT       = "s3://reflective-persistent-prod/<your-username>/pyrosai"
# OUT_DIR         = "./pyrosai_output"
# MODELS          = None
# MAX_MEMBERS     = 3
# BBOX            = None
# RUN_ATTRIBUTION = True
