#!/bin/bash

for c in 0.0
do
	for s in 1 2 3 4 5
	do
		python simulate_networks.py --ratio_inputs_stimulated 0.2 --correlation $c --seed $s
	done
done
