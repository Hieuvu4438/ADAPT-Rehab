"""Dataset loaders for benchmark evaluation."""
import os
import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class VideoSample:
    """A single video sample for evaluation."""
    video_path: str
    exercise_type: str
    person_name: str


def load_yoga_dataset(data_dir: str) -> List[VideoSample]:
    """Load yoga video dataset from directory.

    Expected filename format: {PersonName}_{ExerciseName}.mp4
    """
    video_dir = os.path.join(data_dir, "yoga_datasets", "Yoga_Vid_Collected")
    if not os.path.exists(video_dir):
        print(f"[Dataset] Directory not found: {video_dir}")
        return []

    samples = []
    for filepath in sorted(glob.glob(os.path.join(video_dir, "*.mp4"))):
        filename = os.path.basename(filepath)
        name = os.path.splitext(filename)[0]

        # Parse: PersonName_ExerciseName
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            person, exercise = parts
        else:
            person, exercise = "unknown", name

        samples.append(VideoSample(
            video_path=filepath,
            exercise_type=exercise,
            person_name=person,
        ))

    print(f"[Dataset] Loaded {len(samples)} videos, "
          f"{len(set(s.exercise_type for s in samples))} exercise types")
    return samples


def group_by_exercise(samples: List[VideoSample]) -> dict:
    """Group samples by exercise type."""
    groups = {}
    for s in samples:
        groups.setdefault(s.exercise_type, []).append(s)
    return groups
