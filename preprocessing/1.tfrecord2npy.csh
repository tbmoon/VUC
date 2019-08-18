#!/bin/csh

###########################################################
#
# Your selections.
#
# data_type = 'train', 'valid', 'test'
# which_challenge = '2nd_challenge', '3rd_challenge'
#
###########################################################

set use_all_classes = 'no'
set data_type = 'valid'
set which_challenge = '3rd_challenge'

###########################################################

set n_files = 4000 
#set n_files = 3844 

###########################################################

set i = 0

while ( $i < $n_files )
	@ j = $i + 10

	echo "$i" "$j"
	python tfrecord2npy.py --use_all_classes=$use_all_classes --data_type=$data_type --which_challenge=$which_challenge --start=$i --end=$j

	@ i = $i + 10
end

###########################################################
