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

## Quality and Alias Tooling

Runtime gloss lookup uses `aliases.json` before falling back to fingerspelling.
The alias file is a JSON object keyed by user vocabulary or phrase:

```json
{
  "version": 1,
  "aliases": {
    "mom": "mother",
    "dad": "father",
    "hi": "hello",
    "bye": "goodbye"
  }
}
```

Useful maintenance commands from the repository root:

```bash
python/.venv/bin/python python/lexicon_tools.py weakest --limit 25
python/.venv/bin/python python/lexicon_tools.py quality --limit 25 \
  --output python/lexicon_wlasl/quality_audit.json
python/.venv/bin/python python/lexicon_tools.py unresolved "hi mom" "running classes"
python/.venv/bin/python python/lexicon_tools.py aliases --terms-file user_vocab.txt
python/.venv/bin/python python/lexicon_tools.py suggest "colour" "phone call"
```

`quality_audit.json` is intentionally compact: it stores summary counts plus
the weakest entries and ranked slices for short clips, unstable hand tracking,
pose jumpiness, incomplete coverage, and missing or unreadable pose data.
