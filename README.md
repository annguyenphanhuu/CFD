# CFD

## Preprocessing
python pixels_coords_testing.py --zip_path './30HINH3D.zip' --output_folder './30_case_duyanh'
python pixels_coords_testing.py --zip_path './Case_100.zip' --output_folder './100_case_si'


## Training

### Training U
python U/train.py --trainpath './100_case_si/processed/' --checkpoint False --epocheval 1 --numepoch 10000
--directory 'U/training_U/'

### Training P
python U/train.py --trainpath './100_case_si/processed/' --checkpoint False --epocheval 1 --numepoch 10000
--directory 'P/training_P/'