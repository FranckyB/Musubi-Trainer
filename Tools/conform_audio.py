from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from tkinter import Tk, filedialog, messagebox


TARGET_DURATION_SECONDS = 5.167
OUTPUT_FOLDER_NAME = "conform"
SUPPORTED_EXTENSIONS = {
	".wav",
	".flac",
	".mp3",
	".m4a",
	".ogg",
	".opus",
	".aac",
	".wma",
}


def find_ffmpeg() -> str | None:
	return shutil.which("ffmpeg")


def ask_for_source_folder() -> Path | None:
	root = Tk()
	root.withdraw()
	root.attributes("-topmost", True)
	folder = filedialog.askdirectory(title="Select the folder containing audio files")
	root.destroy()
	if not folder:
		return None
	return Path(folder)


def iter_audio_files(source_dir: Path) -> list[Path]:
	return sorted(
		path
		for path in source_dir.rglob("*")
		if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and OUTPUT_FOLDER_NAME not in path.parts
	)


def conform_audio_file(ffmpeg_executable: str, source_dir: Path, input_path: Path, output_root: Path) -> None:
	relative_path = input_path.relative_to(source_dir)
	output_path = output_root / relative_path
	output_path.parent.mkdir(parents=True, exist_ok=True)

	command = [
		ffmpeg_executable,
		"-y",
		"-i",
		str(input_path),
		"-vn",
		"-af",
		f"apad=pad_dur={TARGET_DURATION_SECONDS}",
		"-t",
		f"{TARGET_DURATION_SECONDS}",
		str(output_path),
	]
	subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
	ffmpeg_executable = find_ffmpeg()
	if ffmpeg_executable is None:
		print("FFmpeg was not found on PATH. Please install FFmpeg and try again.")
		return 1

	source_dir = ask_for_source_folder()
	if source_dir is None:
		print("No folder selected. Exiting.")
		return 1

	audio_files = iter_audio_files(source_dir)
	if not audio_files:
		messagebox.showinfo("Conform Audio", "No supported audio files were found in the selected folder.")
		return 0

	output_root = source_dir / OUTPUT_FOLDER_NAME
	output_root.mkdir(parents=True, exist_ok=True)

	success_count = 0
	failed_files: list[Path] = []

	for audio_path in audio_files:
		try:
			conform_audio_file(ffmpeg_executable, source_dir, audio_path, output_root)
			success_count += 1
			print(f"Conformed: {audio_path}")
		except subprocess.CalledProcessError:
			failed_files.append(audio_path)
			print(f"Failed: {audio_path}")

	summary = f"Conformed {success_count} file(s) into '{OUTPUT_FOLDER_NAME}'."
	if failed_files:
		summary += f"\nFailed: {len(failed_files)}"
	messagebox.showinfo("Conform Audio", summary)

	if failed_files:
		return 1
	return 0


if __name__ == "__main__":
	sys.exit(main())
