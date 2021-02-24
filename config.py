import os
import datetime

# change these!
subreddit = 'dankmemes'
author = 'made by meme-monitor @memeinfluence'
mosiac_height, mosiac_width = 18, 32 # number of tiles 
tile_height, tile_width = 28, 36 # pixel dims for each tile
num_frames = 30
fps = 2
offset = 75 # number of new images introduced each frame

batch_size = 8
num_workers = 1
umap_training_set_size = 1000

# directories
data_dir = 'data/platforms/reddit'
working_dir = os.path.join(data_dir, f'{subreddit}/')
media_dir = os.path.join(data_dir, 'media')
mosaic_dir = os.path.join(working_dir, 'mosaics')
output_dir = os.path.join(working_dir, 'renders')

# files
image_lookup_file = os.path.join(working_dir, 'media.csv.gz')
logits_file = os.path.join(working_dir, 'image_features.csv.gz')
full_metadata_file = os.path.join(working_dir, 'full_metadata.csv.gz')
sample_dataset_file = os.path.join(working_dir, 'sample_media.csv.gz')
two_dim_embeddings_file = os.path.join(working_dir, '2d_embeddings.csv')
file_animation = os.path.join(output_dir,'doppler_mosaic.mp4')

for _dir in [working_dir, media_dir, output_dir, mosaic_dir]:
    os.makedirs(_dir, exist_ok=True)
    
# shared variables
skip_hash = ['NOHASH', '0000000000000000', 'nan']
n_dimensions = 2048 # features from resnet50, change this is you change the model in feature extraction.
cols_conv_feats = [f'conv_{n}' for n in range(n_dimensions)]
