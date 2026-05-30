# WLASL pose lexicon

This directory contains generated `.pose` files for the app's word-level ASL
demo lexicon. The files were built from WLASL video metadata with
`python/build_lexicon.py`; raw WLASL videos are not included.

Source dataset: https://github.com/dxli94/WLASL

WLASL is licensed under the Computational Use of Data Agreement (C-UDA). The
upstream project states that WLASL data is for academic and computational use
only and disallows commercial use. Keep those restrictions in mind when
redistributing this repository or using the generated pose files.

Current generated set:

- 206 indexed ASL glosses
- 206 `.pose` files in `ase/`
- Generated from the most-recorded WLASL glosses, extending past the top 200 to
  replace source videos that were unavailable or below the hand-confidence gate
