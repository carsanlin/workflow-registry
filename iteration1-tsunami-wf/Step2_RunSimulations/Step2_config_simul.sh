#!/bin/bash -e

if [ $# -eq 0 ]; then
    echo "No arguments provided"
    echo "Usage: Step2_launch_simul.sh BS(or PS) grdfile hours"
    exit 1
fi

#initialization
seistype=$1
SCEDIR=Step2_$seistype
if [ ! -d $SCEDIR ]; then
    mkdir $SCEDIR
fi
grid=$2
prophours=$3
nep=$4 # Number of events to be simulated in joint packets (1,4,8,16,32,64,128,256,512)
sims_files=$5

EVENTSFILE=Step1_scenario_list_$seistype\.txt
nscenarios=`cat $EVENTSFILE | wc -l` # Number of scenarios to be simulated
echo 'Number of scenario ' $nscenarios;
nd=`echo "${#nscenarios}"` # nd is the number of digits for the variable events (example for 10->2, for 13536->5)

if [ ! -d logfiles_$seistype ]; then
    mkdir logfiles_$seistype
fi

#preparing jobs
rootname=simulations$seistype
if (( $nscenarios % $nep == 0 )); then
    njobs=$((( $nscenarios / $nep ) ))
else
    njobs=$((( $nscenarios / $nep ) +1 ))
fi
echo 'Number of jobs ' $njobs

jini=1
jfin=$njobs

for j in $(seq $jini $jfin); do
#for j in $(seq $jini 2); do    #######################TEST

    echo "chunk $j / ${jfin} "
    inpfile=$rootname$j\.txt
    touch $inpfile

    s1=$(((( j - 1 ) * $nep) + 1))
    s2=$(( j * $nep ))
    if [ $j -eq $njobs ]; then
        s2=$nscenarios
    fi
    #echo $s1
    #echo $s2
    NN=$((( $s2 - $s1 ) +1 ))
    #echo $s1,$s2,$NN

    for i in $(seq $s1 $s2); do
        #echo $i
        #EVENTNAME=`sed -n ${i}p $EVENTSFILE | awk -F' ' '{print $1}'` #Get the name (first field) of the event
        PARAMS=`sed -n ${i}p $EVENTSFILE`
        sh Step2_simul_$seistype\.sh $PARAMS $grid $prophours $nd

        numsce=$(printf %0$nd\i $i)
        EVENTNAME=$seistype\_scenario$numsce
        #echo $EVENTNAME
        parfile=$SCEDIR/$EVENTNAME/parfile.txt
        echo $parfile >> $inpfile
    done
    echo $inpfile >> $sims_files

    if [ $s1 == 1 ]; then
        one=$(printf %0$nd\i 1)
        ln -s $SCEDIR/$seistype\_scenario$one/parfile.txt parfile1.txt
        ./get_load_balancing parfile1.txt 1 1
        rm parfile1.txt
    fi

done
