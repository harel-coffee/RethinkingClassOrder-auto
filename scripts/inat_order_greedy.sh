# export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:/usr/local/cudnn-7.6/lib64
NUM_EXEMPLARS=20
COMMON_FLAGS='--epochs 70 --dataset inat --base_cl 9 --nb_cl 9 --total_cl 81'

for i in {1..1}; do
  for order in optimal_supervised_wdist_nb_cl_9_beam_0_seed_1993 optimal_ssl_wdist_nb_cl_9_beam_0_seed_1993 optimal_supervised_mdled_nb_cl_9_beam_0_seed_1993 optimal_ssl_mdled_nb_cl_9_beam_0_seed_1993 optimal_semantic_wordnet_wup_nb_cl_9_beam_0_seed_1993; do

    # base model
    python main.py --order_type $order --order_idx $i --memory_type none --bias_rect none --reg_type none --base_model $COMMON_FLAGS
    python main.py --order_type $order --order_idx $i --memory_type none --bias_rect none --reg_type none --base_model --no_final_relu $COMMON_FLAGS

    # iCaRL (CVPR'17)
    python main.py --order_type $order --order_idx $i --memory_type episodic --reg_type lwf --bias_rect none --embedding --test_cosine --num_exemplars $NUM_EXEMPLARS $COMMON_FLAGS

    # eeil (ECCV'18)
    python main.py --order_type $order --order_idx $i --memory_type episodic --reg_type lwf --bias_rect eeil --num_exemplars $NUM_EXEMPLARS $COMMON_FLAGS

    # lsil (CVPR'19)
    python main.py --order_type $order --order_idx $i --memory_type episodic --reg_type lwf --bias_rect bic --bias_aug --adjust_lwf_w --num_exemplars $NUM_EXEMPLARS $COMMON_FLAGS

    # il2m (ICCV'19)
    python main.py --order_type $order --order_idx $i --memory_type episodic --reg_type none --bias_rect il2m --num_exemplars $NUM_EXEMPLARS $COMMON_FLAGS

    # mdfcil (CVPR'20)
    python main.py --order_type $order --order_idx $i --memory_type episodic --reg_type lwf --bias_rect weight_aligning --adjust_lwf_w --num_exemplars $NUM_EXEMPLARS $COMMON_FLAGS

    # post-scaling (CVPRW'21)
    python main.py --order_type $order --order_idx $i --memory_type episodic --reg_type lwf --bias_rect post_scaling --num_exemplars $NUM_EXEMPLARS $COMMON_FLAGS

  done

done
