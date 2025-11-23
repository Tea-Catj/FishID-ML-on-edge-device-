from pathlib import Path
from typing import Optional
from ModelConvert import convert_model


def ensure_nn_archive(nn_archive_path: str, base_dir: Optional[Path] = None) -> str:
    """Ensure the NN archive exists. If not, prompt user for a .pt to convert and return the archive path.

    Returns the resolved archive path as a string.
    Raises FileNotFoundError if user aborts or no archive found.
    """
    if base_dir is None:
        base_dir = Path(__file__).parent

    expected_archive = Path(nn_archive_path)
    if not expected_archive.is_absolute():
        expected_archive = (base_dir / expected_archive).resolve()

    def _find_archive(search_root: Path):
        patterns = ["**/yolo11n*", "**/*.rvc2*", "**/*.tar.xz"]
        for pat in patterns:
            for p in search_root.glob(pat):
                if p.is_file():
                    # ignore .pt matches
                    if p.suffix.lower() == '.pt':
                        continue
                    return p.resolve()
        return None

    if expected_archive.exists():
        return str(expected_archive)

    # try searching workspace
    found = _find_archive(base_dir)
    if found:
        return str(found)

    # Prompt user for .pt path
    def _prompt_for_model(root: Path) -> Path:
        print("No archive found. Please provide the path to the .pt model to convert.")
        print("Enter an absolute path, a path relative to the script, a directory, or a filename pattern. Type 'exit' to abort.")
        while True:
            user_in = input("Model path: ").strip().strip('"').strip("'")
            if not user_in:
                print("Empty input — please provide a path or 'exit'.")
                continue
            if user_in.lower() in ('exit', 'quit'):
                raise FileNotFoundError("User aborted model selection")

            candidate = Path(user_in)
            if not candidate.is_absolute():
                candidate = (root / candidate).resolve()

            if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == '.pt':
                return candidate

            if candidate.exists() and candidate.is_dir():
                pts = list(candidate.glob('**/*.pt'))
                if pts:
                    print(f"Found {len(pts)} .pt files in {candidate}, using {pts[0]}")
                    return pts[0].resolve()
                else:
                    print(f"No .pt files found inside directory {candidate}. Try again.")
                    continue

            matches = list(root.glob(f"**/{user_in}")) + list(root.glob(f"**/{user_in}*.pt"))
            matches = [m for m in matches if m.is_file() and m.suffix.lower() == '.pt']
            if matches:
                print(f"Found matching .pt: {matches[0]}")
                return matches[0].resolve()

            print(f"Could not resolve '{user_in}' to a .pt file. Try again or type 'exit'.")

    # If there is exactly one .pt in the workspace, choose it automatically.
    pt_candidates = list(base_dir.glob('**/*.pt'))
    pt_candidates = [p for p in pt_candidates if p.is_file()]
    if len(pt_candidates) == 1:
        pt_to_convert = pt_candidates[0].resolve()
        print(f"No archive found — automatically selecting the only .pt in workspace: {pt_to_convert}")
    elif len(pt_candidates) > 1:
        print(f"Multiple .pt models found in workspace ({len(pt_candidates)}).")
        for idx, p in enumerate(pt_candidates, start=1):
            print(f"  {idx}: {p}")
        print("Enter the number of the model to convert, a path, or 'exit'.")
        
        # allow the prompt loop to resolve user input
        def _choose_from_list(root: Path) -> Path:
            while True:
                user_in = input("Select model (number/path): ").strip()
                if not user_in:
                    print("Empty input — please select a number, provide a path, or 'exit'.")
                    continue
                if user_in.lower() in ('exit', 'quit'):
                    raise FileNotFoundError("User aborted model selection")
                # if a number, pick from list
                if user_in.isdigit():
                    n = int(user_in)
                    if 1 <= n <= len(pt_candidates):
                        return pt_candidates[n-1].resolve()
                    else:
                        print(f"Invalid selection {n}. Choose 1-{len(pt_candidates)}.")
                        continue
                # otherwise try to resolve as path or glob
                candidate = Path(user_in)
                if not candidate.is_absolute():
                    candidate = (root / candidate).resolve()
                if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == '.pt':
                    return candidate
                # try glob
                matches = list(root.glob(f"**/{user_in}")) + list(root.glob(f"**/{user_in}*.pt"))
                matches = [m for m in matches if m.is_file() and m.suffix.lower() == '.pt']
                if matches:
                    return matches[0].resolve()
                print(f"Could not resolve '{user_in}' to a .pt file. Try again or type 'exit'.")

        pt_to_convert = _choose_from_list(base_dir)
    else:
        pt_to_convert = _prompt_for_model(base_dir)

    print(f"Will attempt to convert model: {pt_to_convert}")
    converted = convert_model(str(pt_to_convert))

    # Try to extract an archive path from the converter return value
    def _extract_path_from_converted(obj):
        candidates = []
        if isinstance(obj, str):
            candidates.append(Path(obj))
        if isinstance(obj, dict):
            for key in ('path', 'archive', 'output', 'file', 'output_path', 'artifact', 'download_path'):
                if key in obj:
                    try:
                        candidates.append(Path(obj[key]))
                    except Exception:
                        pass
        for attr in ('path', 'archive', 'output', 'file', 'output_path', 'artifact', 'save_path', 'download_path'):
            if hasattr(obj, attr):
                try:
                    val = getattr(obj, attr)
                    if isinstance(val, (str, Path)):
                        candidates.append(Path(val))
                except Exception:
                    pass
        for c in candidates:
            try:
                if c.exists():
                    # Ignore paths that point back to the original .pt
                    try:
                        if c.resolve() == pt_to_convert.resolve():
                            continue
                    except Exception:
                        pass
                    # Ignore .pt files returned by the converter (we want an archive)
                    if c.suffix.lower() == '.pt':
                        continue
                    return c.resolve()
            except Exception:
                continue
        return None

    found_from_converted = _extract_path_from_converted(converted)
    if found_from_converted:
        return str(found_from_converted)

    # If converter returned a URL, try to download it
    if isinstance(converted, str) and converted.lower().startswith(('http://', 'https://')):
        try:
            import requests
            print(f"Converter returned a URL; attempting to download: {converted}")
            resp = requests.get(converted, stream=True, timeout=30)
            resp.raise_for_status()
            out_dir = base_dir / 'converted_artifacts'
            out_dir.mkdir(parents=True, exist_ok=True)
            filename = Path(converted).name or f"converted_{pt_to_convert.stem}.tar.xz"
            out_path = out_dir / filename
            with open(out_path, 'wb') as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
            if out_path.exists():
                return str(out_path.resolve())
        except Exception as e:
            print(f"Failed to download converted artifact: {e}")

    # Finally, search for common archive files written into the workspace
    def _find_archive(search_root: Path):
        patterns = ["**/*.rvc2*", "**/*.tar.xz"]
        for pat in patterns:
            for p in search_root.glob(pat):
                if p.is_file():
                    if p.suffix.lower() == '.pt':
                        continue
                    return p.resolve()
        return None

    found_after = _find_archive(base_dir)
    if found_after:
        return str(found_after)

    # If we reach here, conversion did not produce a detectable NN archive.
    # Raise so the caller (Main.py) can decide how to handle the failure.
    raise FileNotFoundError(f"Conversion completed but no NN archive was found for {pt_to_convert}")
    