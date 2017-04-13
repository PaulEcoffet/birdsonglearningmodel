"""Generate all the configuration needed for the cluster."""

from grid_search import get_confs
import argparse
import os
import shutil
import datetime
from os.path import join
import json
from glob import iglob

WALLTIME = {'q1heure': '1:00:00',
            'q12heure': '12:00:00',
            'q1jour': '24:00:00',
            'q1semaine': '167:00:00'}


COMMAND = "/home/paul.ecoffet/.virtualenvs/birdsongs/bin/python "
          + "grid_search.py --single-job -n {name} " \
          + "--outdir={outdir} {conffile}"

SCRIPT = """#!/bin/sh
#PBS -N {name}
#PBS -o log/{name}.out
#PBS -b log/{name}.err
#PBS -l walltime={walltime}
#PBS -l ncpus=1
#PBS -d /home/paul.ecoffet/birdsonglearningmodel/model
{command}
"""


def purge_dir(dir):
    """Remove all the files in `dir`."""
    for fname in iglob(os.path.join(dir, '*')):
        os.remove(fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--name', required=True, type=str)
    ap.add_argument('-q', '--queue', type=str, default='q1jour')
    ap.add_argument('-t', '--tmpdir', type=str, default='tmpdir')
    ap.add_argument('confdir', type=str)
    args = ap.parse_args()
    time_str = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    outdir = join('res', args.name + '_' + time_str)
    os.makedirs(args.tmpdir, exist_ok=True)
    purge_dir(args.tmpdir)
    confspath = args.tmpdir
    to_call = ["#!/bin/sh"]
    for name, conf in get_confs(args.confdir):
        runname = os.path.join(confspath, name)
        with open(runname + '.json', 'w') as f:
            json.dump(conf, f, indent=4)
        command = COMMAND.format(name=name, outdir=outdir,
                                 conffile=runname + '.json')
        with open(runname + '.sh', 'w') as f:
            f.write(SCRIPT.format(name=name, walltime=WALLTIME[args.queue],
                                  command=command))
        to_call.append('qsub ' + join('./', runname + '.sh'))
    with open('run_{}.sh'.format(args.name), 'w') as f:
        f.write('\n'.join(to_call))


if __name__ == '__main__':
    main()
