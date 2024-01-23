#!/bin/sh

#SBATCH -q shared
##SBATCH --qos=debug
#SBATCH --job-name=56Co_sim_SIS2
#SBATCH -C cpu


#SBATCH --image=legendexp/legend-software:latest

#SBATCH -o /global/cfs/cdirs/m2676/users/nfuad/56Co_line_source_sim/Co56_SIS2_neg376_%j.out
#SBATCH -A m2676
#SBATCH -t 5:00:00

BASE_MACRO_DIR=/global/homes/f/fnafis/LEGEND/legend/sims/macros
TEMP_MACRO_DIR=/global/homes/f/fnafis/LEGEND/temp
MPP=/global/homes/f/fnafis/LEGEND/legend/sims/mpp.sh

cp ${BASE_MACRO_DIR}/56Co_SIS2_neg376mm.mac ${TEMP_MACRO_DIR}/56Co_SIS2_neg376mm_${SLURM_JOB_ID}.mac

sed -i "s/neg376mm.*/neg376mm_${SLURM_JOB_ID}.root/g" ${TEMP_MACRO_DIR}/56Co_SIS2_neg376mm_${SLURM_JOB_ID}.mac

shifter bash -c "source /global/homes/f/fnafis/LEGEND/setup_mage.sh;
MaGe ${TEMP_MACRO_DIR}/56Co_SIS2_neg376mm_${SLURM_JOB_ID}.mac;
source ${MPP} ${SLURM_JOB_ID}
"


