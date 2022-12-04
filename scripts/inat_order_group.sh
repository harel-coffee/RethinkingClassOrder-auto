# export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:/usr/local/cudnn-7.6/lib64
NUM_EXEMPLARS=20
COMMON_FLAGS='--dataset inat --base_cl 27 --nb_cl 27 --total_cl 81'

for i in {1..5}; do
  for order in group; do

    # base model
    python main.py $COMMON_FLAGS --order_type $order --order_idx $i --memory_type none --bias_rect none --reg_type none --base_model
    python main.py $COMMON_FLAGS --order_type $order --order_idx $i --memory_type none --bias_rect none --reg_type none --base_model --no_final_relu

    # strong baseline
    python main.py $COMMON_FLAGS --order_type $order --order_idx $i --memory_type episodic --reg_type lwf --bias_rect post_scaling --num_exemplars $NUM_EXEMPLARS

    # iCaRL (CVPR'17)
    python main.py $COMMON_FLAGS --order_type $order --order_idx $i --memory_type episodic --reg_type lwf --bias_rect none --embedding --test_cosine --num_exemplars $NUM_EXEMPLARS

    # eeil (ECCV'18)
    python main.py $COMMON_FLAGS --order_type $order --order_idx $i --memory_type episodic --reg_type lwf --bias_rect eeil --num_exemplars $NUM_EXEMPLARS

    # il2m (ICCV'19)
    python main.py $COMMON_FLAGS --order_type $order --order_idx $i --memory_type episodic --reg_type none --bias_rect il2m --num_exemplars $NUM_EXEMPLARS

    # lsil (CVPR'19)
    python main.py $COMMON_FLAGS --order_type $order --order_idx $i --memory_type episodic --reg_type lwf --bias_rect bic --bias_aug --adjust_lwf_w --num_exemplars $NUM_EXEMPLARS

    # mdfcil (CVPR'20)
    python main.py $COMMON_FLAGS --order_type $order --order_idx $i --memory_type episodic --reg_type lwf --bias_rect weight_aligning --adjust_lwf_w --num_exemplars $NUM_EXEMPLARS

    # post-scaling (CVPRW'21)
    python main.py $COMMON_FLAGS --order_type $order --order_idx $i --memory_type episodic --reg_type lwf --bias_rect post_scaling --num_exemplars $NUM_EXEMPLARS

  done

done
