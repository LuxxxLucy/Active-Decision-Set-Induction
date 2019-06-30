#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --account=def-ester

#SBATCH --mail-user=<jla624@sfu.ca>
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load python/3.6
source ~/ml-interpretation/bin/activate

echo 'Hello, world!'
sleep 30
