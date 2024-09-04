CUDA_VISIBLE_DEVICES=0 python RunSegBatch.py --input_path data/K-Hairstyle/processed_data_0830/Validation/image563 --output_path data/K-Hairstyle/processed_data_0830/Validation/image563_portrait_mask
echo "Success: data/K-Hairstyle/processed_data_0830/Validation/image563"
CUDA_VISIBLE_DEVICES=1 python RunSegBatch.py --input_path data/K-Hairstyle/processed_data_0830/Validation/image1024 --output_path data/K-Hairstyle/processed_data_0830/Validation/image1024_portrait_mask
echo "Success: data/K-Hairstyle/processed_data_0830/Validation/image1024"
CUDA_VISIBLE_DEVICES=2 python RunSegBatch.py --input_path data/K-Hairstyle/processed_data_0830/Training/image563 --output_path data/K-Hairstyle/processed_data_0830/Training/image563_portrait_mask
echo "Success: data/K-Hairstyle/processed_data_0830/Training/image563"
CUDA_VISIBLE_DEVICES=3 python RunSegBatch.py --input_path data/K-Hairstyle/processed_data_0830/Training/image1024 --output_path data/K-Hairstyle/processed_data_0830/Training/image1024_portrait_mask
echo "Success: data/K-Hairstyle/processed_data_0830/Training/image1024"
