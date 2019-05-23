device=3
datasets=""
task="vae"

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
        * )             datasets+=" $1"
    esac
    shift
done

if [ "$seed" == "" ]; then
    echo "| must specify a training seed"
    exit 1
fi

if [ "$datasets" == "" ]; then
    datasets="dsprites cars3d shapes3d"
fi

case $task in
    "vae" )     hparam="beta"
		values="1 4 8 16"
		;;
    "wae" )     hparam="beta"
		values="1 4 8 16"
		;;
    "tc" )      hparam="beta"
		values="1 4 8 16"
		;;
    "factor" )  hparam="gamma"
		values="10 20 40 80"
		;;
    "wtc" )     hparam="gamma"
	        values="1 4 8 16"
		;;
    "wtc_wae" ) hparam="beta"
		values="1 4 8 16"
		;;
    * )        exit 1
esac

for dataset in $datasets
do
    echo "| training $task on $dataset"
    for val in $values
    do
	python train.py \
	       --device-id $device \
	       --no-epoch-checkpoints \
	       --dataset $dataset \
	       --task $task \
	       --max-update 300000 \
	       --seed $seed \
	       --$hparam $val \
	       --save-dir /mnt/bhd/yijunxiao/disent/checkpoints/$dataset/$task/$hparam-$val/$seed \
	       > logs/D-$dataset.T-$task.${hparam^^}-$val.S-$seed.log
    done
    ntfy send "finished training $task on $dataset with seed $seed"
done
ntfy send "all done $task on datasets ($datasets) with seed $seed"
