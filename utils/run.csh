#!/bin/tcsh

################################################################################

#set num_splits = 100
#set num_epochs = 20
#set step_size  = 4

################################################################################

set num_splits = 10
set num_epochs = 10
set step_size  = 3

################################################################################

set which_challenge = '2nd_challenge'

################################################################################

echo "________________________________________________________________________"

@ gamma_power = -1
@ iepoch = 0
while (${iepoch} < ${num_epochs}) 

	
	if (${iepoch} ) % (${step_size} ) == 0 then
		@ gamma_power ++ 
	endif		

	python utils/split_dataframe.py --which_challenge=${which_challenge} --n_splits=${num_splits}

	@ isplit = 0
	while (${isplit} < ${num_splits})

		
		@ cepoch = ${iepoch} + 1
		@ csplit = ${isplit} + 1

		echo "| Epoch [${cepoch}/${num_epochs}] | Split [${csplit}/${num_splits}] |"
		echo "________________________________________________________________________"

		if (${iepoch} > 0 || ${isplit} > 0) then
			python train.py --load_model=True \
							--load_split=True \
							--epoch_number=${iepoch} \
							--split_number=${isplit} \
							--num_epochs=1 \
							--num_splits=${num_splits} \
							--gamma_power=${gamma_power}
		else
			python train.py --load_split=True \
							--epoch_number=${iepoch} \
							--split_number=${isplit} \
							--num_epochs=1 \
							--num_splits=${num_splits} \
							--gamma_power=${gamma_power}
		endif
	
		echo "________________________________________________________________________"

		@ isplit ++

	end

	echo
	@ iepoch ++

end

################################################################################
