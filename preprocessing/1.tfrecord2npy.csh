#!/bin/csh

###########################################################
#
# Your selections.
#
# data_type = 'train', 'valid', 'test'
# which_challenge = '2nd_challenge', '3rd_challenge'
#
###########################################################

set data_type = 'train'
set which_challenge = '2nd_challenge'

###########################################################

set n_files = 500
#set n_files = 3844 

###########################################################

set i = 0

while ( $i < $n_files )
	@ j = $i + 10

	echo "$i" "$j"
	python tfrecord2npy.py --base_dir='/run/media/hoosiki/WareHouse3/mtb/datasets/VU/' --data_type=$data_type --which_challenge=$which_challenge --start=$i --end=$j

	@ i = $i + 10
end

###########################################################
