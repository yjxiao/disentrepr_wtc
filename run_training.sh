device=3
datasets=""
task="vae"
seeds="1 11 42 73 89"    # these are aribitrary

while [ "$1" != "" ]; do
    case $1 in
        -d | --device ) shift
                        device=$1
                        ;;
	-t | --task )   shift
			task=$1
			;;
        * )             datasets+=" $1"
    esac
    shift
done

if [ "$datasets" == "" ]; then
    datasets="dsprites cars3d shapes3d"
fi

if [ "$task" == "all" ]; then
    tasks="vae tc factor wtc mmd_tc"
else
    tasks="$task"
fi

for task in $tasks
do
    case $task in
	"vae" )    hparam="beta"
		   values="1 4 8 16"
		   ;;
	"wae" )    hparam="beta"
		   values="1 4 8 16"
		   ;;
	"tc" )     hparam="beta"
		   values="1 4 8 16"
		   ;;
	"factor" ) hparam="gamma"
		   values="10 20 40 80"
		   ;;
	"wtc" )    hparam="gamma"
		   values="1 4 8 16 40"
		   ;;
	"mmd_tc" ) hparam="gamma"
		   values="10 40 80 160"
		   ;;
	* )        exit 1
    esac
    
    for dataset in $datasets
    do
	echo "| training $task on $dataset"
	if [ "$dataset" == "cars3d" ]
	then
	    lr='1e-5'
	else
	    lr='1e-4'
	fi
	for val in $values
	do
	    for seed in $seeds
	    do
		python train.py \
		       --device-id $device \
		       --no-validate \
		       --no-epoch-checkpoints \
		       --dataset $dataset \
		       --task $task \
		       --lr $lr \
		       --max-update 300000 \
		       --seed $seed \
		       --$hparam $val \
		       --save-dir /mnt/bhd/yijunxiao/disent/checkpoints/$dataset/$task/$hparam-$val/$seed \
		       > logs/D-$dataset.T-$task.${hparam^^}-$val.S-$seed.log
	    done
	done
    done
done
