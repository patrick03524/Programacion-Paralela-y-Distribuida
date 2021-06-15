if [ ! -d asplos_2021_ae ]; then
    echo "clone AE repo"
    git clone https://github.com/brad-mengchi/asplos_2021_ae
fi

cd asplos_2021_ae/benchmarks/src/
source setup_environment
make oop_bench
cd ../../../