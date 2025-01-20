# CFD

python pixels_coords_testing.py --zip_path './DATA/30HINH3D.zip' --output_folder './30_case_duyanh'
python pixels_coords_testing.py --zip_path './DATA/Case_100.zip' --output_folder './100_case_si'

python train.py --trainpath './100_case_si/processed/' --checkpoint False --epocheval 1 --numepoch 10000 --directory 'training_GAN_CFD/'