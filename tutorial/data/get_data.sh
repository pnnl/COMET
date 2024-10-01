prefix="https://suitesparse-collection-website.herokuapp.com/MM/"
declare -a matrices=("HB/bcsstk12.tar.gz" "Williams/cant.tar.gz" "Williams/cop20k_A.tar.gz" "Williams/pdb1HYS.tar.gz" "Bova/rma10.tar.gz" "Hamm/scircuit.tar.gz" "DNVS/shipsec1.tar.gz")

for m in "${matrices[@]}"
do
    curl -L -O $prefix$m
done

for m in $(ls *.gz)
do
    tar -xzvf $m
    rm $m
done

for d in $(ls -d */)
do 
    cd $d
    mv *.mtx ../
    cd ../
    rm -r $d
done