# WLASL pose lexicon

This directory contains generated `.pose` files for the app's word-level ASL
demo lexicon. The first set was built from WLASL video metadata with
`python/build_lexicon.py`; later expansion used preprocessed WLASL MediaPipe
landmarks via `python/import_processed_wlasl.py`. Raw WLASL videos are not
included.

Source dataset: https://github.com/dxli94/WLASL

WLASL is licensed under the Computational Use of Data Agreement (C-UDA). The
upstream project states that WLASL data is for academic and computational use
only and disallows commercial use. Keep those restrictions in mind when
redistributing this repository or using the generated pose files.

Current generated set:

- 2,000 indexed ASL glosses
- 2,000 `.pose` files in `ase/`
- Generated from the most-recorded and highest-priority available WLASL glosses,
  with a focus on common communication, family, food, school/work, health, time,
  questions, and conversation-repair vocabulary
