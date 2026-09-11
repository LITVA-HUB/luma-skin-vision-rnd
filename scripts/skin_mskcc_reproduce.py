"""Run the frozen summary algorithm in a new output root, then audit it."""
import argparse
from pathlib import Path
import skin_mskcc_summary_pilot as pilot
import skin_mskcc_audit as audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory; prior evidence is never overwritten')
    # Only destinations differ. Data, model candidates, seeds, fit budget,
    # validation selection and metric code remain the original frozen code.
    pilot.OUT = args.output / 'report'
    pilot.RUN = args.output / 'models'
    pilot.main()
    audit.main(pilot.OUT, pilot.RUN)


if __name__ == '__main__':
    main()
