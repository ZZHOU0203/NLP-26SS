import argparse
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def resolve_path(path: str) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return BASE_DIR / path_obj


def read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def split_data(src_lines: list[str], tgt_lines: list[str], dev_size: int, test_size: int):
    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"Source/reference length mismatch: {len(src_lines)} vs {len(tgt_lines)}")

    needed = dev_size + test_size
    if len(src_lines) < needed:
        raise ValueError(
            f"Dataset has {len(src_lines)} sentence pairs, but dev+test requires {needed}."
        )

    train_src = src_lines[: -needed]
    train_tgt = tgt_lines[: -needed]
    dev_src = src_lines[-needed: -test_size]
    dev_tgt = tgt_lines[-needed: -test_size]
    test_src = src_lines[-test_size:]
    test_tgt = tgt_lines[-test_size:]

    return {
        "train": (train_src, train_tgt),
        "dev": (dev_src, dev_tgt),
        "test": (test_src, test_tgt),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Create deterministic KDE4 de-hsb splits.")
    parser.add_argument("--src_path", default="../data/KDE4.de-hsb.de")
    parser.add_argument("--ref_path", default="../data/KDE4.de-hsb.hsb")
    parser.add_argument("--output_dir", default="../data/splits")
    parser.add_argument("--dev_size", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=2000)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading raw data...")
    src_lines = read_lines(resolve_path(args.src_path))
    tgt_lines = read_lines(resolve_path(args.ref_path))

    splits = split_data(src_lines, tgt_lines, args.dev_size, args.test_size)

    print("\nWriting deterministic splits...")
    for split_name, (src_split, tgt_split) in splits.items():
        write_lines(output_dir / f"{split_name}.de", src_split)
        write_lines(output_dir / f"{split_name}.hsb", tgt_split)
        print(f"{split_name}: {len(src_split)} sentence pairs")

    print("\nSaved splits to:", output_dir)


if __name__ == "__main__":
    main()
