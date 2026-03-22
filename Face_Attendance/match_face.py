#!/usr/bin/env python3
"""Match face images against a local folder of known faces."""

from __future__ import annotations

import argparse
import io
import json
import logging
import pickle
import shutil
import time
import warnings
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\.",
    category=UserWarning,
    module=r"face_recognition_models(\..*)?$",
)

try:
    import face_recognition
except ImportError as exc:
    face_recognition = None
    _FACE_RECOGNITION_ERROR = exc
else:
    _FACE_RECOGNITION_ERROR = None


NO_MATCH_LABEL = "NO_MATCH"
CACHE_VERSION = 1
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_KNOWN_DIR = Path(__file__).resolve().parent / "known_faces"
DEFAULT_TEST_INPUT_DIR = Path(__file__).resolve().parent / "test_inputs"
DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parent / "already_checked_images"
ENCODING_MODES = {"auto", "detect", "full"}

LOGGER = logging.getLogger("match_face")


def _require_face_recognition() -> None:
    if face_recognition is not None:
        return
    raise RuntimeError(
        "face_recognition is required but not installed. "
        "Install with: pip install face_recognition"
    ) from _FACE_RECOGNITION_ERROR


def _is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _normalize_mode(mode: str) -> str:
    clean_mode = str(mode).strip().lower()
    if clean_mode in ENCODING_MODES:
        return clean_mode
    LOGGER.warning("Unknown encode mode '%s'. Using auto.", mode)
    return "auto"


def _load_image(image_source: str | Path | np.ndarray | bytes) -> Optional[np.ndarray]:
    _require_face_recognition()

    try:
        if isinstance(image_source, np.ndarray):
            image = image_source
        elif isinstance(image_source, bytes):
            image = face_recognition.load_image_file(io.BytesIO(image_source))
        else:
            image = face_recognition.load_image_file(str(image_source))
    except Exception as exc:
        LOGGER.warning("Couldn't read image '%s': %s", image_source, exc)
        return None

    if image is None:
        return None

    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    if image.ndim != 3 or image.shape[2] != 3:
        LOGGER.warning("Unsupported image shape: %s", getattr(image, "shape", None))
        return None

    return image


def _pack_face_rows(locations, encodings, image_ref) -> list[dict[str, Any]]:
    if len(encodings) != len(locations):
        LOGGER.warning(
            "Detected %s faces but got %s encodings for '%s'.",
            len(locations),
            len(encodings),
            image_ref,
        )

    packed = []
    for location, raw_encoding in zip(locations, encodings):
        encoding = np.asarray(raw_encoding, dtype=np.float32)
        if encoding.shape != (128,):
            LOGGER.warning("Unexpected encoding shape for '%s': %s", image_ref, encoding.shape)
            continue
        packed.append({"location": tuple(location), "encoding": encoding})
    return packed


def _encode_detected_faces(image: np.ndarray, image_ref) -> list[dict[str, Any]]:
    try:
        locations = face_recognition.face_locations(image, model="hog")
    except Exception as exc:
        LOGGER.warning("Face detection failed for '%s': %s", image_ref, exc)
        return []

    if not locations:
        LOGGER.warning("No face found in '%s'.", image_ref)
        return []

    try:
        encodings = face_recognition.face_encodings(
            image,
            known_face_locations=locations,
            num_jitters=1,
        )
    except Exception as exc:
        LOGGER.warning("Encoding failed for '%s': %s", image_ref, exc)
        return []

    if not encodings:
        LOGGER.warning("No encoding produced for '%s'.", image_ref)
        return []

    return _pack_face_rows(locations, encodings, image_ref)


def _encode_full_frame(image: np.ndarray, image_ref) -> list[dict[str, Any]]:
    height, width = image.shape[:2]
    full_frame = [(0, width, height, 0)]

    try:
        encodings = face_recognition.face_encodings(
            image,
            known_face_locations=full_frame,
            num_jitters=1,
        )
    except Exception as exc:
        LOGGER.warning("Encoding failed for '%s': %s", image_ref, exc)
        return []

    if not encodings:
        LOGGER.warning("No encoding produced for '%s'.", image_ref)
        return []

    encoding = np.asarray(encodings[0], dtype=np.float32)
    if encoding.shape != (128,):
        LOGGER.warning("Unexpected encoding shape for '%s': %s", image_ref, encoding.shape)
        return []

    return [{"location": full_frame[0], "encoding": encoding}]


def encode_faces(
    image_path: str | Path | np.ndarray | bytes,
    mode: str = "auto",
) -> list[dict[str, Any]]:
    image = _load_image(image_path)
    if image is None:
        return []

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        LOGGER.warning("Image has invalid size: (%s, %s)", height, width)
        return []

    mode = _normalize_mode(mode)
    if mode == "detect":
        return _encode_detected_faces(image, image_path)
    if mode == "full":
        return _encode_full_frame(image, image_path)

    detected = _encode_detected_faces(image, image_path)
    if detected:
        return detected

    return _encode_full_frame(image, image_path)


def encode_face(
    image_path: str | Path | np.ndarray | bytes,
    mode: str = "auto",
) -> Optional[np.ndarray]:
    faces = encode_faces(image_path, mode=mode)
    if not faces:
        return None

    def area(face: dict[str, Any]) -> int:
        top, right, bottom, left = face["location"]
        return max(1, (bottom - top) * (right - left))

    return max(faces, key=area)["encoding"]


def _load_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        return {}

    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        LOGGER.warning("Couldn't load cache '%s': %s. Rebuilding.", cache_path, exc)
        return {}

    if not isinstance(payload, dict):
        LOGGER.warning("Invalid cache format in '%s'. Rebuilding.", cache_path)
        return {}

    return payload


def _iter_images(folder: Path):
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and _is_supported_image(path)),
        key=lambda path: path.name.lower(),
    )


def _build_unique_destination_path(destination_dir: Path, file_name: str) -> Path:
    destination = destination_dir / file_name
    if not destination.exists():
        return destination

    stem = Path(file_name).stem
    suffix = Path(file_name).suffix
    stamp = int(time.time() * 1000)
    counter = 1

    while True:
        candidate = destination_dir / f"{stem}_{stamp}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def load_or_build_db(
    known_dir: str | Path,
    cache_path: str | Path = "encodings_cache.pkl",
    encode_mode: str = "auto",
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    _require_face_recognition()

    known_dir_path = Path(known_dir).resolve()
    cache_file = Path(cache_path).resolve()
    encode_mode = str(encode_mode).strip().lower()

    if encode_mode not in ENCODING_MODES:
        raise ValueError(
            f"Invalid encode_mode '{encode_mode}'. Expected one of: {sorted(ENCODING_MODES)}"
        )

    if not known_dir_path.exists():
        raise FileNotFoundError(f"Known directory does not exist: {known_dir_path}")
    if not known_dir_path.is_dir():
        raise NotADirectoryError(f"Known path is not a directory: {known_dir_path}")

    cached = _load_cache(cache_file)
    cached_records = {}
    cache_matches = (
        cached.get("version") == CACHE_VERSION
        and cached.get("known_dir") == str(known_dir_path)
        and cached.get("encoding_mode") == encode_mode
        and isinstance(cached.get("records"), dict)
    )
    if cache_matches:
        cached_records = cached["records"]

    current_files = list(_iter_images(known_dir_path))
    current_keys = {str(path.resolve()) for path in current_files}

    records: dict[str, dict[str, Any]] = {}
    reused_count = 0
    rebuilt_count = 0
    skipped_count = 0

    for image_path in current_files:
        key = str(image_path.resolve())

        try:
            mtime_ns, size = _file_signature(image_path)
        except OSError as exc:
            LOGGER.warning("Cannot stat '%s': %s. Skipping.", image_path, exc)
            skipped_count += 1
            continue

        label = image_path.stem
        cached_record = cached_records.get(key)
        can_reuse = (
            isinstance(cached_record, dict)
            and cached_record.get("mtime_ns") == mtime_ns
            and cached_record.get("size") == size
            and cached_record.get("label") == label
            and isinstance(cached_record.get("encoding"), np.ndarray)
            and cached_record["encoding"].shape == (128,)
        )

        if can_reuse:
            records[key] = cached_record
            reused_count += 1
            continue

        encoding = encode_face(image_path, mode=encode_mode)
        if encoding is None:
            skipped_count += 1
            continue

        records[key] = {
            "path": key,
            "label": label,
            "mtime_ns": mtime_ns,
            "size": size,
            "encoding": encoding.astype(np.float32),
        }
        rebuilt_count += 1

    removed_files = sorted(set(cached_records) - current_keys)

    ordered_records = sorted(records.values(), key=lambda item: Path(item["path"]).name.lower())
    labels = [record["label"] for record in ordered_records]
    if ordered_records:
        encodings_np = np.vstack([record["encoding"] for record in ordered_records]).astype(
            np.float32
        )
    else:
        encodings_np = np.empty((0, 128), dtype=np.float32)

    meta = {
        "known_dir": str(known_dir_path),
        "encoding_mode": encode_mode,
        "cache_path": str(cache_file),
        "num_known_files": len(current_files),
        "num_valid_encodings": int(encodings_np.shape[0]),
        "reused_from_cache": reused_count,
        "rebuilt_encodings": rebuilt_count,
        "skipped_files": skipped_count,
        "removed_files": removed_files,
        "files": {
            record["path"]: {
                "label": record["label"],
                "mtime_ns": record["mtime_ns"],
                "size": record["size"],
            }
            for record in ordered_records
        },
    }

    cache_payload = {
        "version": CACHE_VERSION,
        "known_dir": str(known_dir_path),
        "encoding_mode": encode_mode,
        "records": {record["path"]: record for record in ordered_records},
    }

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("wb") as handle:
            pickle.dump(cache_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        LOGGER.warning("Failed to save cache '%s': %s", cache_file, exc)
        meta["cache_saved"] = False
    else:
        meta["cache_saved"] = True

    return encodings_np, labels, meta


def _empty_match(threshold: float) -> dict[str, Any]:
    return {
        "match_label": NO_MATCH_LABEL,
        "match_distance": None,
        "threshold": float(threshold),
        "top_k": [],
        "status": NO_MATCH_LABEL,
    }


def match_face(
    input_encoding: Optional[np.ndarray],
    db_encodings: np.ndarray,
    db_labels: Sequence[str],
    threshold: float = 0.6,
    top_k: int = 5,
) -> dict[str, Any]:
    result = _empty_match(threshold)

    if input_encoding is None or input_encoding.shape != (128,):
        return result
    if db_encodings.size == 0 or not db_labels:
        return result
    if db_encodings.ndim != 2 or db_encodings.shape[1] != 128:
        LOGGER.error("Invalid db_encodings shape: %s", db_encodings.shape)
        return result
    if len(db_labels) != db_encodings.shape[0]:
        LOGGER.error(
            "Mismatched db sizes: %s encodings vs %s labels",
            db_encodings.shape[0],
            len(db_labels),
        )
        return result

    distances = np.linalg.norm(db_encodings - input_encoding.reshape(1, 128), axis=1)
    ranked = np.argsort(distances)
    limit = max(0, min(int(top_k), len(ranked)))

    result["top_k"] = [
        {"label": str(db_labels[idx]), "distance": float(distances[idx])}
        for idx in ranked[:limit]
    ]

    if len(ranked) == 0:
        return result

    best_idx = int(ranked[0])
    best_distance = float(distances[best_idx])
    result["match_distance"] = best_distance

    if best_distance <= float(threshold):
        result["match_label"] = str(db_labels[best_idx])
        result["status"] = "MATCH"

    return result


def match_faces(
    input_faces: Sequence[dict[str, Any]],
    db_encodings: np.ndarray,
    db_labels: Sequence[str],
    threshold: float = 0.6,
    top_k: int = 5,
) -> dict[str, Any]:
    faces_output = []
    present_labels = []
    seen_labels = set()

    for face_index, face in enumerate(input_faces):
        single = match_face(
            face.get("encoding"),
            db_encodings,
            db_labels,
            threshold=threshold,
            top_k=top_k,
        )
        label = str(single.get("match_label", NO_MATCH_LABEL))
        matched = single.get("status") == "MATCH"

        if matched and label != NO_MATCH_LABEL and label not in seen_labels:
            seen_labels.add(label)
            present_labels.append(label)

        faces_output.append(
            {
                "face_index": face_index,
                "location": list(face.get("location")) if face.get("location") is not None else None,
                "match_label": label,
                "match_distance": single.get("match_distance"),
                "status": single.get("status", NO_MATCH_LABEL),
                "present": bool(matched),
                "top_k": single.get("top_k", []),
            }
        )

    return {
        "status": "PRESENT" if present_labels else NO_MATCH_LABEL,
        "threshold": float(threshold),
        "num_faces_detected": len(input_faces),
        "num_present": len(present_labels),
        "present_labels": present_labels,
        "faces": faces_output,
    }


def _run_match_for_input(
    image_path: str | Path,
    db_encodings: np.ndarray,
    db_labels: Sequence[str],
    threshold: float,
    top_k: int,
    encoding_mode: str,
    multi_face: bool,
) -> dict[str, Any]:
    if multi_face:
        faces = encode_faces(image_path, mode=encoding_mode)
        return match_faces(faces, db_encodings, db_labels, threshold, top_k)

    encoding = encode_face(image_path, mode=encoding_mode)
    return match_face(encoding, db_encodings, db_labels, threshold, top_k)


def _format_simple_output(result: dict[str, Any], multi_face: bool) -> str:
    if multi_face:
        present_labels = result.get("present_labels", [])
        if present_labels:
            return ",".join(str(label) for label in present_labels)
        return NO_MATCH_LABEL

    if result.get("status") == "MATCH":
        return str(result.get("match_label", NO_MATCH_LABEL))

    return NO_MATCH_LABEL


def _watch_loop(args: argparse.Namespace) -> int:
    input_dir = Path(args.watch_dir).resolve()
    processed_dir = Path(args.processed_dir).resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    poll_interval = max(0.1, float(args.poll_interval))
    min_file_age = max(0.0, float(args.min_file_age))
    max_cycles = max(0, int(args.max_cycles))
    cycle_count = 0

    LOGGER.info("Watching input directory: %s", input_dir)
    LOGGER.info("Processed files directory: %s", processed_dir)

    try:
        while True:
            cycle_count += 1
            now = time.time()
            ready_files = []

            for image_path in _iter_images(input_dir):
                try:
                    age_seconds = now - image_path.stat().st_mtime
                except OSError:
                    continue
                if age_seconds >= min_file_age:
                    ready_files.append(image_path)

            if ready_files:
                db_encodings, db_labels, _meta = load_or_build_db(
                    args.known_dir,
                    args.cache_path,
                    encode_mode=args.encoding_mode,
                )

                for image_path in ready_files:
                    try:
                        result = _run_match_for_input(
                            image_path,
                            db_encodings,
                            db_labels,
                            args.threshold,
                            args.top_k,
                            args.encoding_mode,
                            args.multi_face,
                        )
                    except Exception as exc:
                        result = {
                            "status": "ERROR",
                            "error": str(exc),
                            "input_file": str(image_path),
                        }
                    finally:
                        destination = _build_unique_destination_path(processed_dir, image_path.name)
                        try:
                            shutil.move(str(image_path), str(destination))
                        except Exception as move_exc:
                            LOGGER.error(
                                "Failed to move processed file '%s' to '%s': %s",
                                image_path,
                                destination,
                                move_exc,
                            )

                    if args.json:
                        print(
                            json.dumps(
                                {
                                    "input_file": str(image_path),
                                    "processed_file": str(destination),
                                    "result": result,
                                },
                                ensure_ascii=True,
                            )
                        )
                    else:
                        if result.get("status") == "ERROR":
                            print(f"{image_path.name}: ERROR")
                        else:
                            print(f"{image_path.name}: {_format_simple_output(result, args.multi_face)}")

            if max_cycles and cycle_count >= max_cycles:
                return 0

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Match face image(s) against a known face directory."
    )
    parser.add_argument(
        "--known_dir",
        default=str(DEFAULT_KNOWN_DIR),
        help=f"Directory of known cropped face images. Default: {DEFAULT_KNOWN_DIR}",
    )
    parser.add_argument(
        "--input",
        help="Input image path for one-off matching. Required unless --watch is used.",
    )
    parser.add_argument(
        "--cache_path",
        default="encodings_cache.pkl",
        help="Path to cache file (pickle). Default: encodings_cache.pkl",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Distance threshold for a positive match. Default: 0.6",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of nearest candidates to include. Default: 5",
    )
    parser.add_argument(
        "--encoding_mode",
        default="auto",
        choices=sorted(ENCODING_MODES),
        help="Face encoding mode for known and input images. Default: auto",
    )
    parser.add_argument(
        "--multi_face",
        action="store_true",
        help="Detect and match all faces in the input image.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch an input folder, process new images, and move them after checking.",
    )
    parser.add_argument(
        "--watch_dir",
        default=str(DEFAULT_TEST_INPUT_DIR),
        help=f"Input folder watched when --watch is enabled. Default: {DEFAULT_TEST_INPUT_DIR}",
    )
    parser.add_argument(
        "--processed_dir",
        default=str(DEFAULT_PROCESSED_DIR),
        help=f"Folder where checked images are moved when --watch is enabled. Default: {DEFAULT_PROCESSED_DIR}",
    )
    parser.add_argument(
        "--poll_interval",
        type=float,
        default=1.0,
        help="Seconds between watch polls. Default: 1.0",
    )
    parser.add_argument(
        "--min_file_age",
        type=float,
        default=1.0,
        help="Minimum file age in seconds before processing in watch mode. Default: 1.0",
    )
    parser.add_argument(
        "--max_cycles",
        type=int,
        default=0,
        help="Watch mode only: stop after N poll cycles. 0 means run forever.",
    )
    parser.add_argument(
        "--log_level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level. Default: WARNING",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON result instead of simple text output.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    if args.watch:
        return _watch_loop(args)

    if not args.input:
        output = {"status": "ERROR", "error": "--input is required unless --watch is enabled."}
        if args.json:
            print(json.dumps(output, ensure_ascii=True))
        else:
            print("ERROR")
        return 1

    try:
        db_encodings, db_labels, _meta = load_or_build_db(
            args.known_dir,
            args.cache_path,
            encode_mode=args.encoding_mode,
        )
        output = _run_match_for_input(
            args.input,
            db_encodings,
            db_labels,
            args.threshold,
            args.top_k,
            args.encoding_mode,
            args.multi_face,
        )
    except Exception as exc:
        output = _empty_match(args.threshold)
        output["status"] = "ERROR"
        output["error"] = str(exc)
        if args.json:
            print(json.dumps(output, ensure_ascii=True))
        else:
            print("ERROR")
        return 1

    if args.json:
        print(json.dumps(output, ensure_ascii=True))
    else:
        print(_format_simple_output(output, args.multi_face))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
