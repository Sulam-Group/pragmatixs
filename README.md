To make `evaluate.py` work, you need to manually edit the source code of the package `open_clip`
- install a version > 0.30.0
- Go to `open_clip/transformer.py`, line 251, and then set `need_weights=True`.