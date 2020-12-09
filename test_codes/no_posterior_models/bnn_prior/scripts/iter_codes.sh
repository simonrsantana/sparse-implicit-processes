mkdir exe
mkdir exe/tmp
for i in 0.0001 1.0 ; do 
	mkdir exe/$i
	cat alpha_script.sh | sed -e s/ALPHA/$i/g -e s/DATA/"bim_data.txt"/g > exe/tmp/alpha_$i.sh
	cat exe/tmp/alpha_$i.sh | sed -e  s/SPLIT/0/g > exe/$i/alpha_split_0.sh ; 
done
rm exe/tmp -r
