#!/bin/bash

for c in 0.0
do
	python infer_weights.py --ratio_inputs_stimulated 0.2 --correlation $c --seed 1
done
