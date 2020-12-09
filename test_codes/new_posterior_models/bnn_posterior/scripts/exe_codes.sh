for i in 0.0001 1.0; do
        sbatch -A ada2_serv -p cccmd exe/$i/alpha_split_0.sh ;
done

