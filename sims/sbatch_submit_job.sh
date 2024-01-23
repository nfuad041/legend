#!/bin/sh

#SBATCH -q shared
##SBATCH --qos=debug
#SBATCH --job-name=56Co_sim_SIS2
#SBATCH -C cpu


#SBATCH --image=legendexp/legend-software:latest

#SBATCH -o /global/cfs/cdirs/m2676/users/nfuad/56Co_line_source_sim/Co56_SIS2_neg376_%j.out
#SBATCH -A m2676
#SBATCH -t 5:00:00

cp /global/homes/f/fnafis/MAGE_FILES/56Co_SIS2_neg376mm.mac /global/homes/f/fnafis/MAGE_FILES/56Co_SIS2_neg376mm_${SLURM_JOB_ID}.mac

sed -i "s/neg376mm.*/neg376mm_${SLURM_JOB_ID}.root/g" /global/homes/f/fnafis/MAGE_FILES/56Co_SIS2_neg376mm_${SLURM_JOB_ID}.mac

shifter bash -c "source /global/homes/f/fnafis/MAGE_FILES/setup_mage.sh;
MaGe /global/homes/f/fnafis/MAGE_FILES/56Co_SIS2_neg376mm_${SLURM_JOB_ID}.mac;
source /global/cfs/cdirs/m2676/users/nfuad/LEGEND/SIMS/mpp.sh ${SLURM_JOB_ID} > /global/cfs/cdirs/m2676/users/nfuad/LEGEND/SIMS/mpp.sh.${SLURM_JOB_ID}.out;
echo '/global/cfs/cdirs/m2676/users/nfuad/LEGEND/SIMS/56Co_SIS2_neg376mm_${SLURM_JOB_ID}_evt.root' | cat /global/cfs/cdirs/m2676/users/nfuad/LEGEND/SIMS/evt_filenames.txt;
"


