import os
import threading #runs things in parallell
import numpy as np #handles arrays, used for image processing
import rasterio #reads big .tif images efficiently
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

FILE_PATH       = "file_example_TIFF_10MB.tiff"
TILE_SIZE_X     = 1024
TILE_SIZE_Y     = 512        
OVERLAP         = 64           
MAX_WORKERS     = min(6, os.cpu_count()) #parallell threads based on CPU cores
SAVE_TO_DISK    = True         
OUTPUT_DIR      = Path("tiles")
MAX_IN_FLIGHT   = MAX_WORKERS * 3   #cap queued tasks to control RAM
_thread_local = threading.local() #each thread gets its own rasterio handle, avoids thread-safety issues

def get_ds(): #opens .tif file once/thread, then reuses it
    """Return this thread's open rasterio dataset (opened once, reused)."""
    if not hasattr(_thread_local, "ds"):
        _thread_local.ds = rasterio.open(FILE_PATH)
    return _thread_local.ds

# Tile coordinate generator
def iter_tile_coords(width, height):
    """
    Generates (col, row, w, h) for every tile position in the image.
    - Slides a window across image in 960 pixel steps, producing overlap    
    - For each position it clamps to image bounds so edge tiles don't go out of range
    - Duplicate coordinates (from clamping) are skipped via a seen.
    """
    step_y = TILE_SIZE_Y - OVERLAP #512-64 = 448 pixels/step 
    step_x = TILE_SIZE_X - OVERLAP #512-64 = 448 pixels/step 
    seen = set() #ensures no repeated tiles

    for y in range(0, height, step_y): 
        for x in range(0, width, step_x):
            row_pos = min(y, max(0, height - TILE_SIZE_Y))
            column_pos = min(x, max(0, width  - TILE_SIZE_X)) #prevents tiles from going outside the image


            if (column_pos, row_pos) in seen:
                continue
            seen.add((column_pos, row_pos))

            #Actual tile size (edge tiles may be smaller)
            tile_width = min(TILE_SIZE_X, width  - column_pos)
            tile_height = min(TILE_SIZE_Y, height - row_pos)
            yield column_pos, row_pos, tile_width, tile_height #(x, y, width, height)

def read_tile(column_pos, row_pos, tile_width, tile_height): #(x,y,width, height)
    """
    Read one tile using rasterio's window and return a uint8 numpy array (H, W, C).
    Returns None if the tile is background.
    """
    ds   = get_ds()  
    data = ds.read(window=rasterio.windows.Window(column_pos, row_pos, tile_width, tile_height))   
    tile_data  = np.moveaxis(data, 0, -1)  #(C, H, W) → (H, W, C)

    #Normalise to uint8 if needed
    if tile_data.dtype == np.uint16: 
        tile_data = (tile_data / 256).astype(np.uint8)  #scale 0-65535 to 0-255
    elif tile_data.dtype != np.uint8:   #if other -> scale to 0-255 
        low_color, high_color = tile_data.min(), tile_data.max()
        tile_data = ((tile_data - low_color) / (high_color - low_color + 1e-9) * 255).astype(np.uint8)
    
    return tile_data

#Worker function (runs in each thread) 
def process_tile(column_pos, row_pos, tile_width, tile_height):
    """
    Read tile, check background, optionally save, return result dict.
    Returns None for background or failed tiles.
    """
    try:
        tile_data = read_tile(column_pos, row_pos, tile_width, tile_height)
        if tile_data is None:
            return None  #if background tile, skip

        if SAVE_TO_DISK:
            OUTPUT_DIR.mkdir(exist_ok = True)
            Image.fromarray(tile_data).save(
                OUTPUT_DIR / f"tile_y_{row_pos:06d}_x_{column_pos:06d}.png",
                compress_level=1
            )

        return {"y_cords": column_pos, "x_cords": row_pos, "tile": tile_data} #position + image

    except Exception as e:
        print(f"  Tile ({column_pos},{row_pos}) failed: {e}")
        return None

def run_pipeline():
    """
    Tile the whole image using a thread pool.
    Returns a list of dicts: [{"y_cords", "x_cords", "tile"}, ...]
    """
    #Open once just to read image dimensions/size
    with rasterio.open(FILE_PATH) as ds:
        width, height = ds.width, ds.height

    tile_positions = list(iter_tile_coords(width, height))
    total_number_of_tiles = len(tile_positions)
    print(f"Image   : {width} x {height} px")
    print(f"Tiles   : {total_number_of_tiles}  ({TILE_SIZE_X}px x {TILE_SIZE_Y}px, {OVERLAP}px overlap)")
    print(f"Workers : {MAX_WORKERS}")
    

    results = []
    skipped = 0
    done    = 0

    #Thread pool to process tiles in parallel with a memory brake
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool: 
        futures = []
        for column_pos, row_pos, tile_width, tile_height in tile_positions:
            futures.append(pool.submit(process_tile, column_pos, row_pos, tile_width, tile_height)) #each tile becomes a task

            # Memory brake: drain batch before queuing more
            # Prevents thousands of tile arrays accumulating in RAM
            if len(futures) >= MAX_IN_FLIGHT:
                for fut in as_completed(futures):
                    res = fut.result()
                    done += 1
                    if res is None:
                        skipped += 1
                    else:
                        results.append(res)
                futures = []
                if done % 1000 == 0:
                    print(f"  Progress: {done}/{total_number_of_tiles} | skipped {skipped}")
        #Drain any remaining futures after the loop
        for fut in as_completed(futures): #add valid tiles, count skipped ones
            res = fut.result()
            done += 1
            if res is None:
                skipped += 1
            else:
                results.append(res)
    kept = total_number_of_tiles - skipped
    print(f"\nTotal tiles   : {total_number_of_tiles}")
    print(f"Background    : {skipped}  ({skipped/total_number_of_tiles*100:.1f}%)")
    print(f"Tissue tiles  : {kept}  ({kept/total_number_of_tiles*100:.1f}%)")
    print(f"Ready to send : {len(results)} tiles")
    return results

tiles = run_pipeline()


"""___BACKGROUND FILTER SETTINGS___
MIN_TISSUE_FRAC = 0.05 #keep tiles with at least this fraction of content 
BG_WHITE        = 230  #pixels above this = white background
BG_BLACK        = 10   #pixels below this = black background

#Background filter 
def is_background(tile_data):
    
    Return True if the tile is mostly empty (white or black background).
    tile_data shape: (H, W, 3) uint8
    - White background: pixels > BG_WHITE
    - Black background: pixels < BG_BLACK
    A tile needs MIN_TISSUE_FRAC of pixels that are NEITHER white nor black.
    
    gray        = tile_data.mean(axis=2)          #converts RGB to grayscale
    not_white   = np.mean(gray < BG_WHITE)  #fraction that is not white/bg
    not_black   = np.mean(gray > BG_BLACK)  #fraction that is not black/bg
    tissue_frac = min(not_white, not_black) #must pass both checks
    return tissue_frac < MIN_TISSUE_FRAC #if less than 5% real content → skip tile

     #___Lägnst ner i read_tile() ___
    if is_background(tile_data):
        return None
    
    mitten a run pipeline
    print(f"Tissue  : keeping tiles with at least {MIN_TISSUE_FRAC*100:.0f}% tissue\n")
"""
