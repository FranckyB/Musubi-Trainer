# Musubi-Trainer

Musubi-Trainer is a Windows desktop launcher for dataset management and queued LoRA jobs using Musubi-Tuner.
Musubi-Tuner is not included in this repository. You point Musubi-Trainer to an existing Musubi-Tuner folder in Settings.

The app is now job-first:

- Datasets are source assets only
- Jobs are the trainable queue items
- Each job has its own settings, cache, output, and progress metadata

## Current Support

- Model families: Klein (FLUX.2), Krea2, MiniMax H3, LTX, Wan, Z-Image, Qwen
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

- Python 3.11 required
- Unified dependencies from `requirements.txt` (Trainer + Tuner helpers)
- Musubi-Tuner checkout in a separate folder (for example `D:/Musubi-Tuner`)

## Setup (Single venv)

Set up the shared app/training Python environment:

```bat
Setup.bat
```

On Linux:

```bash
./Setup.sh
```

`Setup.bat` attempts to auto-download a matching SageAttention wheel from:
https://github.com/sdbds/SageAttention-for-windows/releases

If auto-download fails, download a matching `.whl` manually from that releases page
and run Setup with `--sage-wheel` pointing to the downloaded file.

`Setup.sh` follows the same shared-venv flow for Linux, skips the Windows-only
`triton-windows` dependency, and tries to install the standard `sageattention`
pip package automatically. If that optional install fails, Setup continues and
you can rerun it with `--sage-wheel` pointing to a compatible local wheel.

On Linux, the visual launcher also needs the system Tk runtime for `tkinter`.
On CachyOS/Arch, install it with:

```bash
sudo pacman -S tk
```


## Launch

From repository root:

```bat
Launch.bat
```

On Linux:

```bash
./Launch.sh
```

`Launch.bat` prefers `venv\Scripts\pythonw.exe` then `venv\Scripts\python.exe`.

`Launch.sh` runs the app with `venv/bin/python` and reports if the shared venv has not been created yet.

## First-Time Setup

1. Open Settings in the app.
2. Set Musubi-Tuner directory (or use the startup prompt to clone/use an existing checkout).
3. Verify model files:
	- Klein Model
	- Klein VAE
	- Klein Text Encoder
4. Save Settings.

For MiniMax H3 specifically, you must set all four MiniMax paths in Settings:

- MiniMax H3 FL2VA/T2VA DiT
- MiniMax H3 Video VAE
- MiniMax H3 Audio VAE
- Qwen3-VL 32B MiniMax-H3 Text Encoder

MiniMax audio-only dataset preparation also requires `ffmpeg` to be available in PATH.

## MiniMax H3 Notes

MiniMax H3 support is available in the launcher, but it is still a heavier and less-forgiving path than Klein/LTX.

Current launcher behavior:

- Video datasets are used directly.
- Image datasets are trained through MiniMax's image-compatible path.
- Audio-only datasets are converted into synthetic black-frame `.mp4` files inside the job folder so MiniMax can ingest them.
- Editing a MiniMax job re-syncs those synthetic `.mp4` files: stale ones are removed, new audio clips are converted, and changed clips are refreshed.

Important limits and expectations:

- MiniMax jobs are generated with `batch_size = 1`.
- Audio-only MiniMax jobs currently use `target_frames = [124]` with `frame_extraction = "head"`.
- At 24 fps, that means each training sample uses about the first `5.17s` of audio/video from a clip, not the full source duration.
- Very long clips are therefore not especially useful for MiniMax in the current launcher flow; shorter clips around that range are a better fit.
- MiniMax H3 has a high VRAM footprint. Even on a `32 GB` GPU, `512x512` with LoRA rank `32` may still run out of memory.

If MiniMax H3 OOMs:

- Lower LoRA `network_dim` / `network_alpha` first. A safer starting point is `16 / 16`.
- Keep resolution conservative.
- Use the job context menu option `Force Recache (Clear Cached Latents)` if you changed source clips and want a clean recache.
- If the job still does not fit, use a lighter family for the same dataset.

MiniMax-specific note:

- Torch compile is disabled automatically for MiniMax jobs in this launcher because Inductor compile-time benchmarking can cause extra VRAM spikes.

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
