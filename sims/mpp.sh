#!/bin/sh

echo processing $1

python3 ~/LEGEND/mage-post-proc/mage-post-proc/L200Apps/process_L200_hit.py /global/cfs/cdirs/m2676/users/nfuad/gamma_3009_sim/sis2_24mm_$1.root -o /global/cfs/cdirs/m2676/users/nfuad/gamma_3009_sim/sis2_24mm_$1_hit.root > ~/LEGEND/temp/$1_hit.out

python3 ~/LEGEND/mage-post-proc/mage-post-proc/L200Apps/process_L200_evt.py /global/cfs/cdirs/m2676/users/nfuad/gamma_3009_sim/sis2_24mm_$1_hit.root -o /global/cfs/cdirs/m2676/users/nfuad/gamma_3009_sim/sis2_24mm_$1_evt.root -c /global/cfs/cdirs/m2676/sims/prodenv/l200a/v1.0.0/inputs/simprod/config/tier/evt/l200a/l200-p03-r000-phy-build_evt.json -d /global/cfs/cdirs/m2676/sims/prodenv/l200a/v1.0.0/inputs/hardware/detectors/germanium/diodes > ~/LEGEND/temp/$1_evt.out

echo "MaGe file --> hit --> evt done for 56Co_SIS2_24mm_$1"
