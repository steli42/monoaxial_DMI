for i in $(find $1 -path "*.json"); do
    echo "Submitted config: $i"
    # ./run.sh $i
    sbatch aion_serial.sh $i
done


# running like ./submit.sh will look for all the json files in the kd_tree folder
# running like ./submit.sh configs passes configs to $1 so it will only inspect the configs dir
# if no config folder exists, then it will just parse the default.json