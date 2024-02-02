#!/bin/sh

echo processing $2

python3 ~/LEGEND/mage-post-proc/mage-post-proc/L200Apps/process_L200_hit.py /global/cfs/cdirs/m2676/users/nfuad/56Co_sim/$1_$2.root -o /global/cfs/cdirs/m2676/users/nfuad/56Co_sim/$1_$2_hit.root > ~/LEGEND/temp/$1_$2_hit.out

python3 ~/LEGEND/mage-post-proc/mage-post-proc/L200Apps/process_L200_evt.py /global/cfs/cdirs/m2676/users/nfuad/56Co_sim/$1_$2_hit.root -o /global/cfs/cdirs/m2676/users/nfuad/56Co_sim/$1_$2_evt.root -c /global/cfs/cdirs/m2676/sims/prodenv/l200a/v1.0.0/inputs/simprod/config/tier/evt/l200a/l200-p03-r000-phy-build_evt.json -d /global/cfs/cdirs/m2676/sims/prodenv/l200a/v1.0.0/inputs/hardware/detectors/germanium/diodes > ~/LEGEND/temp/$1_$2_evt.out

echo "MaGe file --> hit --> evt done for $1_$2"
