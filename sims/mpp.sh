#!/bin/sh

echo processing$1now

python3 ~/legend/legend-swdev-scripts/mage-post-proc/mage-post-proc/L200Apps/process_L200_hit.py ~/56Co_SIS2_neg376mm_$1.root -o ~/56Co_SIS2_neg376mm_$1_hit.root > ~/$1_hit.out

python3 ~/legend/legend-swdev-scripts/mage-post-proc/mage-post-proc/L200Apps/process_L200_evt.py ~/56Co_SIS2_neg376mm_$1_hit.root -o ~/56Co_SIS2_neg376mm_$1_evt.root -c /global/cfs/cdirs/m2676/sims/prodenv/l200a/v1.0.0/inputs/simprod/config/tier/evt/l200a/l200-p03-r000-phy-build_evt.json -d /global/cfs/cdirs/m2676/sims/prodenv/l200a/v1.0.0/inputs/hardware/detectors/germanium/diodes > ~/$1_evt.out

echo "MaGe file --> hit --> evt done for 56Co_SIS2_neg376mm_$1"
