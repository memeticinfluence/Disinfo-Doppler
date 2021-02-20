import os
import numpy as np
from rasterfairy import transformPointCloud2D
from PIL import Image, ImageFont, ImageDraw 
from tqdm import tqdm



def generate_mosaic(embeddings, images, mosaic_width, mosaic_height,
                    tile_width=72, tile_height=56, title= "Doppler Mosaic",
                    title_rbg=(229, 229, 229), save_as_file=False, 
                    return_image=True, verbose=False):
    '''
    Transforms 2-dimensional embeddings to a grid. 
    Plots the images for each embedding in the corresponding grid (mosaic).
    Includes arguments for the dimensions of each tile and the the mosaic.
    '''
    # assign to grid
    grid_assignment = transformPointCloud2D(embeddings, 
                                            target=(mosaic_width, 
                                                    mosaic_height))
    full_width = tile_width * mosaic_width
    full_height = tile_height * (mosaic_height + 2)
    aspect_ratio = float(tile_width) / tile_height
    
    # create an empty image for the mosaic
    mosaic = Image.new('RGB', (full_width, full_height))
    
    # iterate through each image and where it is possed to live.
    for f_img, (idx_x, idx_y) in tqdm(zip(images, grid_assignment[0]), 
                                      disable = not verbose):
        # Find exactly where the image will be
        x, y = tile_width * idx_x, tile_height * idx_y
        
        # read the image, center crop the image and add it to the mosaic
        try:
            img = Image.open(f_img).convert('RGBA')
            tile = resize_image(img, tile_width, tile_height, aspect_ratio)
            mosaic.paste(tile, (int(x), int(y)))
        except Exception as e:
            print(f"Failed to add image {f_img} see error:\n{e}")    
    
    # write an annotation
    #fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', int(tile_height * 1.2) )
    draw = ImageDraw.Draw(mosaic)
    draw.text((4, (tile_height * (mosaic_height)) + 10), 
               title, title_rbg)
    
    if save_as_file and not os.path.exists(save_as_file):
        try:
            mosaic.save(save_as_file)
        except Exception as e:
            print(f'Saving the mosaic to {save_as_file} failed, see error:\n{e}')
    
    if return_image:
        return mosaic
    
def scatterplot_images(embeddings, images,
                       width = 1200, height = 900, 
                       max_dim = 40):
    '''
    Plots images in a scatterplot where coordinates are from
    embeddings.
    '''
    tx, ty = embeddings[:,0], embeddings[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    full_image = Image.new('RGB', 
                           size=(width, height), 
                           color=(55, 61, 71))

    for f_img, x, y in tqdm(zip(images, tx, ty)):
        # read and resize image
        tile = Image.open(f_img)
        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile_width = int(tile.width / rs)
        tile_height = int(tile.height / rs)
        aspect_ratio = float(tile_width) / tile_height 
        tile = resize_image(tile, 
                            tile_width, 
                            tile_height, 
                            aspect_ratio)

        # add the image to the graph               
        x_coord = int((width - max_dim) * x)
        y_coord = int((height - max_dim) * y)
        img_coords = (x_coord, y_coord)
        full_image.paste(tile, 
                         box=img_coords, 
                         mask=tile.convert('RGBA'))
    return full_image

