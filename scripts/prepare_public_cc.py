import argparse
from pathlib import Path

from luma_skin_vision.cc.data import prepare_cube, prepare_sony

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=128)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
output = root / "data/processed/cc128"
rows = prepare_cube(root / "data/public/cube", output, args.size)
prepare_sony(root / "data/public/intel_tau_c5_sony_pilot", output, args.size)
print(f"Prepared {len(rows)} Cube images and 30 external Sony examples")
