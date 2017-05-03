from glob import iglob
from os.path import join, isdir, basename
import sys
from joblib import Parallel, delayed
from syllable_cluster import syllables_from_run


def run(run_name, force):
    try:
        syllables_from_run(run_name, force)
    except FileNotFoundError:
        pass


def main():
    try:
        gridpath = sys.argv[2]
    except IndexError:
        print("USAGE: syllables_extractor.py NB_CPU PATH")
        raise
    run_paths = sorted(
            [run_path for run_path in iglob(join(gridpath, '*'))
             if isdir(run_path)])
    print(run_paths)
    Parallel(n_jobs=int(sys.argv[1]), verbose=10)([
        delayed(run)(run_name, force=True)
        for run_name in run_paths])


if __name__ == '__main__':
    main()
