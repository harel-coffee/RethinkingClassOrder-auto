# export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:/usr/local/cudnn-7.6/lib64

python cosa.py --dist_type wdist --feat_type ssl
python cosa.py --dist_type wdist --feat_type supervised
python cosa.py --dist_type mdled --feat_type ssl
python cosa.py --dist_type mdled --feat_type supervised
python cosa.py --semantic