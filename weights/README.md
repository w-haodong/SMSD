# Weights

Place downloaded model checkpoints in this folder before running inference or evaluation.

Expected structure:

- `weights/fh_data_bs/latest_model.pth`
- `weights/fh_data_lc/latest_model.pth`

Example:

```bash
python main.py --phase predict --image_dir path/to/images --checkpoint weights/fh_data_bs/latest_model.pth
```
