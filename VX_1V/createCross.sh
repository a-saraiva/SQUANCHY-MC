#!/bin/bash
for jj in 0 10 20 30 40 50
do
	mkdir cross$jj
	cd cross$jj
	awk 'NR==5 {$0='$jj'} { print }' ../input.dat > input.dat
		for ii in {1..5}
		do 
			mkdir run$ii
			cd run$ii
			ln -s ../../../PIMC.py pimc.py
			ln -s ../../V_HRL_Grid.npz V_HRL_Grid.npz 
			cp ../input.dat input.dat
			cd ..
		done
	cd ..
done
