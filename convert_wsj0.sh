#!/bin/bash

KALDI_ROOT=~/kaldi # path to kaldi
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe # path to sph2pipe

cd ../../corpus/WSJ0/; # path to wsj0 with SPHERE file
ch1=0
ch2=0

for i in $(seq 15)
do
	cd 11-$i.1/wsj0
	# folder=$(ls)
	for folder in `ls`; do
		if [[ ( "$folder" = "si_"* ) || ( "$folder" = "sd_"* ) ]]
		then 
      echo 11-$i.1/wsj0/$folder
      cd $folder
      for subfold in `ls`; do
        echo "=> processing $subfold"
        cd $subfold
        if [ ! -d ~/corpus/wsj0-wav/$folder/$subfold ]; then
          mkdir -p ~/corpus/wsj0-wav/$folder/$subfold
          echo "make ~/corpus/wsj0-wav/$folder/$subfold"
        fi
        ((ch1+=`ls | grep .wv1 | wc -l`))
        ((ch2+=`ls | grep .wv2 | wc -l`))

        for sound in `ls | grep .wv1`; do
          wavfile=${sound::-3}wav
          $sph2pipe -f wav $sound ~/corpus/wsj0-wav/$folder/$subfold/$wavfile
        done

        cd ..
        
      done
      cd ..
		fi
	done
	cd ../..
done

echo "channel 1 : $ch1"
echo "channel 2 : $ch2"