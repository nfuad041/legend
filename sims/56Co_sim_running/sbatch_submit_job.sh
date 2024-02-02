#!/bin/sh

#SBATCH -q shared
##SBATCH --qos=debug
#SBATCH --job-name=56Co_sim
#SBATCH -C cpu


#SBATCH --image=legendexp/legend-software:latest

#SBATCH -o /global/homes/f/fnafis/LEGEND/temp/$1_%j.out
#SBATCH -A m2676
#SBATCH -t 3:00:00

BASE_MACRO_DIR=/global/homes/f/fnafis/LEGEND/legend/sims/56Co_sim_running
TEMP_MACRO_DIR=/global/homes/f/fnafis/LEGEND/temp

cp ${BASE_MACRO_DIR}/$1.mac ${TEMP_MACRO_DIR}/$1_${SLURM_JOB_ID}.mac
# 1 is the first argument to the script, $1.mac will be the macro file
sed -i "s/sim.root/$1_${SLURM_JOB_ID}.root/g" ${TEMP_MACRO_DIR}/$1_${SLURM_JOB_ID}.mac

MPP=/global/homes/f/fnafis/LEGEND/legend/sims/56Co_sim_running/mpp.sh

shifter bash -c "source /global/homes/f/fnafis/LEGEND/setup_mage.sh;
MaGe ${TEMP_MACRO_DIR}/$1_${SLURM_JOB_ID}.mac;
source ${MPP} $1 ${SLURM_JOB_ID}
"
