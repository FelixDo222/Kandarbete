import tifffile
import numpy as np
import os

######## SELECT INPUT ########
tile_size = 512
overlap = 64
input_file = "25FU1231.tif" # name of input file
tiles_output = "tiles_output" #name of output folder
##############################

def tile_tiff(input_path, output_dir, tile_size, overlap):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with tifffile.TiffFile(input_path) as tif:
        image = tif.asarray(out='memmap')
        height, width = image.shape[:2]
    
    stride = tile_size - overlap 


    
    tile_count = 0
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            tile = image[y : y + tile_size, x : x + tile_size]
            tile_filename = f"tile_{tile_count}_y{y}_x{x}.tif"
            tifffile.imwrite(os.path.join(output_dir, tile_filename), tile)
            
            tile_count += 1


    #summary of pipline
    print(f"--- TIFF Importer Pipeline Summary ---")
    print(f"Input File: {input_path}")
    print(f"Image Dimensions: {width}x{height} pixels")
    print(f"Image Dimensions padded: {width}x{height} pixels")
    print(f"Tiling Parameters: {tile_size}px tiles with {overlap}px overlap")
    print(f"Outcome: Successfully generated {tile_count} tiles in '{output_dir}'.")
    print(f"---------------------------------------")



tile_tiff(input_file, tiles_output, tile_size=tile_size, overlap=overlap)