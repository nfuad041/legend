#!/bin/sh

#SBATCH -q shared
##SBATCH --qos=debug
#SBATCH --job-name=56Co_sim_SIS2
#SBATCH -C cpu


#SBATCH --image=legendexp/legend-software:latest

#SBATCH -o /global/cfs/cdirs/m2676/users/nfuad/56Co_line_source_sim/Co56_SIS2_neg276_%j.out
#SBATCH -A m2676
#SBATCH -t 4:00:00

BASE_MACRO_DIR=/global/homes/f/fnafis/LEGEND/legend/sims/macros
TEMP_MACRO_DIR=/global/homes/f/fnafis/LEGEND/temp
MPP=/global/homes/f/fnafis/LEGEND/legend/sims/mpp.sh

cp ${BASE_MACRO_DIR}/one_gamma_source.mac ${TEMP_MACRO_DIR}/one_gamma_source_${SLURM_JOB_ID}.mac

sed -i "s/24mm.*/24mm_${SLURM_JOB_ID}.root/g" ${TEMP_MACRO_DIR}/one_gamma_source_${SLURM_JOB_ID}.mac

shifter bash -c "source /global/homes/f/fnafis/LEGEND/setup_mage.sh;
MaGe ${TEMP_MACRO_DIR}/one_gamma_source_${SLURM_JOB_ID}.mac;
source ${MPP} ${SLURM_JOB_ID}
"
