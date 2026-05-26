# Musubi-Trainer

Musubi-Trainer is a desktop launcher for running dataset prep, cache generation, and LoRA training with Musubi-Tuner.
Musubi-Tuner is not included in this repository. You point Musubi-Trainer to your existing Musubi-Tuner folder from Settings.

## Current Support

- Model family: Klein
- Platform focus: Windows

## Current Stage Note

- This launcher is still Klein-only at this stage.
- Training parameters are currently hard-coded to a profile that has tested well in my runs.
- Planned to add model selection as well as preset selections.
- Datasets has to be created before hand. This tool only offers the option of creating .txt with preset keyword if not present.

## Highlights

- Dataset card UI with thumbnails and drag-to-reorder persistence
- Queue multiple datasets in one run
- Automatic readiness checks for prep and cache steps
- Step-by-step logging with consistent sections
- Continue queue when one dataset fails (failure summary at end)
- In-progress Start button with click-to-cancel confirmation
- LoRA Post-Hoc EMA merge actions and standalone merge tool
- Settings persistence in src/settings.json

## Folder Layout

Expected training layout under this repo:

- Training/<DatasetName>/images
- Training/<DatasetName>/cache
- Training/<DatasetName>/output
- Training/<DatasetName>/dataset.toml

## Requirements

- Python 3.10+ recommended
- App dependencies from requirements.txt (Pillow, tkinterdnd2)
- Musubi-Tuner checkout in a separate folder (for example D:/Musubi-Tuner)
- Musubi-Tuner virtual environment with required dependencies installed

## App Setup (venv)

From repository root (first time, or after dependency changes):

```bat
Setup.bat
```

This creates a local `venv` and installs app dependencies from `requirements.txt`.

## Launch

From repository root:

```bat
Launch.bat
```

`Launch.bat` prefers `venv\Scripts\pythonw.exe` / `venv\Scripts\python.exe` first.

## First-Time Setup

1. Open Settings in the app.
2. Set Musubi-Tuner directory.
3. Verify model files:
	- Klein Model
	- Klein VAE
	- Klein Text Encoder
4. Save Settings.

The app can auto-detect Musubi Python from:

- <musubi_dir>/venv/Scripts/python.exe
- <musubi_dir>/.venv/Scripts/python.exe

You can also set a manual Python path in Settings.

## Training Workflow

1. Select one or more dataset cards.
2. Press START.
3. The app runs the selected queue and logs each step:
	- Dataset Check
	- Cache Latent
	- Cache Text Encoder
	- Train

When a step is already complete, it is logged as skipped instead of silently omitted.

If one dataset fails, the queue continues with the next dataset.

## LoRA Merge Drag-and-Drop

- The standalone LoRA merge tool supports dragging `.safetensors` files onto the LoRAs list or Merge order list.
- This uses `tkinterdnd2`, installed by `Setup.bat` through `requirements.txt`.

## Cancel Behavior

- During a run, START changes to In Progress (Press to Cancel).
- Clicking it prompts for confirmation.
- Confirming cancels the active process and stops remaining queued datasets.

## Logging and Debugging

- The training step logs the exact command sent to flux_2_train_network.py.
- Cache steps log ready/generated counts.
- End-of-run summary indicates success or failed dataset names.

## CLI Mode (Optional)

You can still run from CLI:

```bat
python -m src.app DatasetA DatasetB
```

Optional flags:

- --prep-dataset
- --cache-latents
- --cache-text
- --train

If no step flags are supplied, all steps are enabled.
