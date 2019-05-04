#!/usr/bin/env bash

# lists files in folder tf_lite_models and compute the time they need to predict an image based on the tensorflow-lite
# benchmark tool, the order in which we make measurements in randomized, repeteaded 2 times and we sleep for some time
# between each run to avoid performance drops that would be dur to processor heating

# function to shuffle the tf_lite_files array, taken from
# https://stackoverflow.com/questions/5533569/simple-method-to-shuffle-the-elements-of-an-array-in-bash-shell
shuffle_tf_lite_files() {
   local i tmp size max rand

   # $RANDOM % (i+1) is biased because of the limited range of $RANDOM
   # Compensate by using a range which is a multiple of the array size.
   size=${#tf_lite_files[*]}
   max=$(( 32768 / size * size ))

   for ((i=size-1; i>0; i--)); do
      while (( (rand=$RANDOM) >= max )); do :; done
      rand=$(( rand % (i+1) ))
      tmp=${tf_lite_files[i]} tf_lite_files[i]=${tf_lite_files[rand]} tf_lite_files[rand]=${tmp}
   done
}

# computes the time needed by the file with the benchmark too and add it to the results
add_results_to_dict() {
    for tf_lite_file in "${tf_lite_files[@]}" ; do
        sleep 10s
        score=$(/home/pi/tf-lite/benchmark_model --graph=${tf_lite_file} --num_runs=50 |& tr -d '\n' | awk '{print $NF}')
        # tr removes the \n and awk gets the last element of the outputs message, |& is used before tr because we want
        # to pipe stderr and not stdout
        echo "${tf_lite_file}: ${score}mus"
       results_dict[${tf_lite_file}]=$(bc -l <<< "${results_dict[${tf_lite_file}]} + ${score}")
    done
}


### script:
# list files and declare dict
tf_lite_files=( $(find tf_lite_models/*))
declare -A results_dict

# fills dictionary
for tf_lite_file in "${tf_lite_files[@]}" ; do
   results_dict[${tf_lite_file}]=0
done

# shuffle array and benchmarks
passes=(1 2 3)
num_passes=${#passes[*]}

for var in "${passes[@]}" ; do
    echo "pass ${var}"
    shuffle_tf_lite_files
    add_results_to_dict
done

# write results
for tf_lite_file in "${tf_lite_files[@]}" ; do
   res=$(bc -l <<< "scale=4; ${results_dict[${tf_lite_file}]} / ${num_passes} / 1000000")  # divide by number of passes 
   # and by one million because output given in ms
   echo "${tf_lite_file}: ${res}" >> results.txt
done
