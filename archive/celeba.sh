echo "Process /data_new/chence/CelebA/Img/img_celeba.7z/meta_1-3.json"
CUDA_VISIBLE_DEVICES=1 python 02.front_view.py -i /data_new/chence/CelebA/Img/img_celeba.7z/meta_1-3.json
echo "Process /data_new/chence/CelebA/Img/img_celeba.7z/meta_2-3.json"
CUDA_VISIBLE_DEVICES=1 python 02.front_view.py -i /data_new/chence/CelebA/Img/img_celeba.7z/meta_2-3.json
echo "Process /data_new/chence/CelebA/Img/img_celeba.7z/meta_3-3.json"
CUDA_VISIBLE_DEVICES=1 python 02.front_view.py -i /data_new/chence/CelebA/Img/img_celeba.7z/meta_3-3.json
