# export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:/usr/local/cudnn-7.6/lib64
COMMON_FLAGS='--dataset inat'

for j in {1..5}; do
  for i in 9; do
    for order in even group; do
      for dist in mdled wdist; do
        python trans_calc.py $COMMON_FLAGS --order_idx $j --dist_type $dist --feat_type ssl --order_type $order --base_cl $i --nb_cl $i
        python trans_calc.py $COMMON_FLAGS --order_idx $j --dist_type $dist --feat_type supervised --order_type $order --base_cl $i --nb_cl $i
      done
      python trans_calc.py $COMMON_FLAGS --order_idx $j --semantic --semantic_type wordnet_wup --order_type $order --base_cl $i --nb_cl $i
    done
  done
done
