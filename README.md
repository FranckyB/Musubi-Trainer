# Musubi-Trainer

Musubi-Trainer is a Windows desktop launcher for dataset management and queued LoRA jobs using Musubi-Tuner.
Musubi-Tuner is not included in this repository. You point Musubi-Trainer to an existing Musubi-Tuner folder in Settings.

The app is now job-first:

- Datasets are source assets only
- Jobs are the trainable queue items
- Each job has its own settings, cache, output, and progress metadata

## Current Support

- Model families: Klein (FLUX.2), LTX, Wan, Z-Image, Qwen
- Tested in this launcher: Klein and LTX
- Platform focus: Windows

## Current Stage Note

- Multi-family support is now available.
- Ongoing validation and quality-of-life updates continue across families.
- Datasets should be prepared first (images). The app can auto-create missing caption `.txt` files.

## Highlights

- Dataset card UI with thumbnails
- Built-in dataset caption editor with live autosave
- Queue-based job system with per-job settings
- Preset system for saving and reusing job configurations
- Per-family preferred preset support for faster Create Job setup
- Job statuses: `queued`, `running`, `paused`, `resume`, `done`, `failed`, `broken`
- Resume-aware step tracking using explicit `progress.json` metadata
- Edit jobs in place and re-evaluate status from current outputs + new target steps
- Queue continues after failures and shows end-of-run summary
- In-progress Start button with click-to-cancel confirmation
- LoRA Post-Hoc EMA merge actions (job context + standalone merge tool)
- Settings persistence in `src/settings.json`

## How It Works

1. Create or import datasets under `Datasets`.
2. Edit captions from dataset cards:
   - Double-click a dataset card to open the caption editor.
   - Right-click a dataset card and choose `Edit Dataset`.
3. Caption text is auto-saved to matching `.txt` files (created automatically when missing).
4. Create one or more jobs from datasets.
5. Configure job-specific training options (steps, optimizer, learning rate, dim/alpha, flags), then save as a preset if desired.
6. Reorder queue, pause/enable jobs, then press START QUEUE.
7. App runs prep/cache/train per runnable job.
8. Resume data and recorded progress determine whether a job is `resume` or `done`.

## Queue Behavior

- `done` jobs are locked from re-enable by checkbox toggle.
- Editing a job re-scans it to determine the correct status.
- If target steps are increased later, a previously completed job can become `resume`.
- `Fix LoRA Names` appears only for `broken` jobs.
- Extra merged files no longer force `broken` when expected job artifacts are present.

## Folder Layout

Expected workspace layout under this repo:

- Datasets/<DatasetName>/... source images and captions
- Jobs/_order.json queue order metadata
- Jobs/<JobName>/settings.json persisted job settings
- Jobs/<JobName>/dataset.toml generated per-job dataset config
- Jobs/<JobName>/progress.json recorded completed steps
- Jobs/<JobName>/cache job cache artifacts
- Jobs/<JobName>/output training outputs
- Jobs/<JobName>/output/merged post-hoc merged outputs

## Requirements

- Python 3.10+ recommended
- App dependencies from `requirements.txt` (`Pillow`, `tkinterdnd2`)
- Musubi-Tuner checkout in a separate folder (for example `D:/Musubi-Tuner`)
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

`Launch.bat` prefers `venv\Scripts\pythonw.exe` then `venv\Scripts\python.exe`.

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

## Job Workflow

1. Create or select a dataset card.
2. Press Create Job.
3. Tune job settings and save.
4. Optionally save those settings as a preset and reuse or reload them for future jobs.
5. Repeat to build queue.
6. Press START QUEUE.
7. The app logs each step for each runnable job:
   - Dataset Check
   - Cache Latent
   - Cache Text Encoder
   - Train

When a step is already complete, it is logged as skipped instead of silently omitted.

If one job fails, the queue continues with the next runnable job.

## LoRA Merge Drag-and-Drop

- The standalone LoRA merge tool supports dragging `.safetensors` files onto the LoRAs list or Merge order list.
- Job context merge writes outputs to `Jobs/<JobName>/output/merged`.
- This uses `tkinterdnd2`, installed by `Setup.bat` through `requirements.txt`.

## Cancel Behavior

- During a run, START QUEUE changes to In Progress (Press to Cancel).
- Clicking it prompts for confirmation.
- Confirming cancels the active process and stops remaining queued jobs.

## Logging and Debugging

- The training step logs the exact command sent to flux_2_train_network.py.
- Cache steps log ready/generated counts.
- End-of-run summary indicates successful and failed job names.

## CLI Mode (Optional)

You can still run from CLI for direct execution:

```bat
python -m src.app DatasetA DatasetB
```

Optional flags:

- --prep-dataset
- --cache-latents
- --cache-text
- --train

If no step flags are supplied, all steps are enabled.
