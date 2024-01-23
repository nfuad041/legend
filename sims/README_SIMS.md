## How I'm running simulations
### Install MGDO+MaGe+MPP
1. git clone legend-swdev-scripts
2. stay out of the legend-swdev-scripts directory, run 'python3 legend-swdev-scripts/installMaGe.py --mgdofork nfuad041 --magefork nfuad041 --mppfork nfuad041 install'
This will create a lot of new files, including setup_mage.sh. You have to run this everytime before running mpp: source setup_mage.sh
3.  'source /global/homes/f/fnafis/LEGEND/legend/sims/submit_job_multiple.sh' to run multiple jobs on NERSC. <-- This runs 'source sbatch_submit_job.sh' N times <-- This i. copies the macro from BASE_MACRO_DIR to a temp dir, then changes the output raw rootfile names with the SLURM_JOB_ID + ii. sources the setup_mage.sh file + iii. runs the macro with MaGe
***4. The base macro file (at /global/homes/f/fnafis/LEGEND/legend/sims/macro/) contains info about how many primaries to run and what the output file name (RAW file only)should be. You can change values there. 
***5. The mpp file (/global/homes/f/fnafis/LEGEND/legend/sims/mpp.sh) runs the post processing - it takes the raw file input (DEF check if that's correct), converts it to a hit and evt file in the same directory with same name with '_hit' and '_evt' appended to the name, using post-processing files from 'mage-post-proc'. Again, DEF check all the params. 
* you can check status of jobs with 'squeue -u <username>' and cancel jobs with 'scancel <jobid>'
