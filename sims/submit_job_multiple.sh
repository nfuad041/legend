#!/bin/sh

## 1M source decays take ~4 hours
N=20
for i in $(seq 1 $N);
do
    sbatch sbatch_submit_job.sh
done
