#feat_dim =(1 2)
for i in `seq 1 32`;
do
    python3 test_spmm_kernel.py --dataset "reddit" --feat ${i} >> res_ran.txt
    sleep 5

    python3 test_spmm_kernel.py --dataset "reddit" --feat ${i} >> res_ran.txt
    sleep 5 

done    
