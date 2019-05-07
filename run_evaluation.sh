device=0
all_datasets="dsprites cars3d shapes3d"
all_tasks="vae tc factor wtc mmd_tc wae"
all_metrics="unsup mig factor"
seed="1"    # these are aribitrary
datasets=""
task=""

while [ "$1" != "" ]; do
    case $1 in
        -d | --device ) shift
                        device=$1
                        ;;
        -t | --task )   shift
                        task=$1
                        ;;
	-s | --seed )   shift
			seed=$1
			;;
	-m | --metric ) shift
			metrics=$1
			;;
        * )             datasets+=" $1"
    esac
    shift
done

if [ "$metrics" == "" ]; then
    metrics="$all_metrics"
fi

if [ "$datasets" == "" ]; then
    datasets="$all_datasets"
fi

if [ "$task" == "all" ]; then
    tasks="$all_tasks"
elif [ "$task" == "" ]; then
    tasks="$all_tasks"
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

    for metric in $metrics
    do
	if [ "$metric" == "factor" ]; then
	    args="--train-batches 10000 --eval-batches 5000 --ve 200"
	else
	    args="--eval-batches 200"
	fi
     
	for dataset in $datasets
	do
	    echo "| evaluating $task on $dataset ($metric)"
	    for val in $values
	    do
		for d in /mnt/bhd/yijunxiao/disent/checkpoints/$dataset/$task/$hparam-$val/* ;
		do
		    if [ -d ${d} ]; then
			python evaluate.py \
			       --device-id $device \
		    	       --task $task \
			       --dataset $dataset \
		    	       --metric $metric \
		    	       --path $d/checkpoint_last.pt \
			       $args \
		    	       --save-results    
		    fi
		done
	    done
	done
    done
done
