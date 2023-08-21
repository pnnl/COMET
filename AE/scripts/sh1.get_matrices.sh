#!/bin/bash

# Under AE/

mkdir data
cd data

matrices=("Bova/rma10" "Williams/cant" "Hamm/scircuit" "Williams/cop20k_A" "HB/bcsstk17" \
          "Williams/pdb1HYS" "Williams/consph" "DNVS/shipsec1" "SNAP/com-Orkut" "SNAP/com-LiveJournal")

for m in "${matrices[@]}"; do
  url="https://suitesparse-collection-website.herokuapp.com/MM/${m}.tar.gz"
  echo "downloading from" $url
  curl -O -L $url
  name=$(echo $m | cut -d'/' -f 2)
  tar xvf "${name}.tar.gz"
  rm -rf "${name}.tar.gz"
done

# cd ..
