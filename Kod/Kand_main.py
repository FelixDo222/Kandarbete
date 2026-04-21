import os
import threading #runs things in parallell
import numpy as np #handles arrays, used for image processing
import rasterio #reads big .tif images efficiently
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

FILE_PATH       = "data/file_example_TIFF_10MB.tiff"
TILE_SIZE_X     = 1280
TILE_SIZE_Y     = 720
OVERLAP         = 64       
MAX_WORKERS     = min(6, os.cpu_count()) #parallell threads based on CPU cores
SAVE_TO_DISK    = True         
OUTPUT_DIR      = Path("tile")
MAX_IN_FLIGHT   = MAX_WORKERS * 3   #cap queued tasks to control RAM
thread_local = threading.local() #each thread gets its own rasterio handle, avoids thread-safety issues

def get_dataset(): #opens .tif file once/thread, then reuses it
    """Return this thread's open rasterio dataset (opened once, reused)."""
    if not hasattr(thread_local, "ds"):
        thread_local.ds = rasterio.open(FILE_PATH)
    return thread_local.ds

# Tile coordinate generator
def iter_tile_coords(image_width, image_height):
    """
    Generates (col, row, w, h) for every tile position in the image.
    - Slides a window across image in 960 pixel steps, producing overlap    
    - For each position it clamps to image bounds so edge tiles don't go out of range
    - Duplicate coordinates (from clamping) are skipped via a seen.
    """
    step_y = TILE_SIZE_Y - OVERLAP
    step_x = TILE_SIZE_X - OVERLAP
    seen = set() #ensures no repeated tiles
    for y in range(0, image_height, step_y):
        for x in range(0, image_width, step_x):
            y_pos = min(y, max(0, image_height - TILE_SIZE_Y))
            x_pos = min(x, max(0, image_width  - TILE_SIZE_X))

            if (x_pos, y_pos) in seen:
                continue
            seen.add((x_pos, y_pos))

            tile_len_x = min(TILE_SIZE_X, image_width  - x_pos)
            tile_len_y = min(TILE_SIZE_Y, image_height - y_pos)

            yield x_pos, y_pos, tile_len_x, tile_len_y




"""___Background Filter___"""

MIN_TISSUE_FRAC = 0.05 #keep tiles with at least this fraction of content 
BG_WHITE        = 230  #pixels above this = white background
BG_BLACK        = 10   #pixels below this = black background

def is_background(tile_data):
    """
    Return True if the tile is mostly empty (white or black background).
    arr shape: (H, W, 3) uint8
    - White background: pixels > BG_WHITE
    - Black background: pixels < BG_BLACK
    A tile needs MIN_TISSUE_FRAC of pixels that are NEITHER white nor black.
    """
    gray        = tile_data.mean(axis=2)          #converts RGB to grayscale
    not_white   = np.mean(gray < BG_WHITE)  #fraction that is not white/bg
    not_black   = np.mean(gray > BG_BLACK)  #fraction that is not black/bg
    tissue_frac = min(not_white, not_black) #must pass both checks
    return tissue_frac < MIN_TISSUE_FRAC #if less than 5% real content → skip tile





def read_tile(x_pos, y_pos, tile_len_x, tile_len_y): #(x,y,width,height)
    """
    Read one tile using rasterio's window and return a uint8 numpy array (H, W, C).
    Returns None if the tile is background.
    """
    ds   = get_dataset()  #dataset, image but not loaded into RAM
    tile_data_raw = ds.read(window=rasterio.windows.Window(x_pos, y_pos, tile_len_x, tile_len_y))   #only loads a small part of the huge image, data.shape = (channels, height, width), (3, 512, 512)
    tile_data = np.moveaxis(tile_data_raw, 0, -1)  #moves channels from position 0 to the end,converts from (C, H, W) → (H, W, C)

    #Normalise to uint8 if needed
    if tile_data.dtype == np.uint16:
        tile_data = (tile_data / 257).astype(np.uint8)  #16-bit (0-65535) → 8-bit (0-255), 257 = 65535/25
    elif tile_data.dtype != np.uint8:   #only uint8/uint16 expected, if not crash early. uint8 already in 0-255, pass through
        low, high = tile_data.min(), tile_data.max()
        tile_data = ((tile_data - low) / (high - low + 1e-9) * 255).astype(np.uint8)
    
    if tile_data.shape[2] == 4:
        tile_data = tile_data[:, :, :3]
        
    "___INKLUDERA ENDAST DENNA KOD OM DU SKA TA BORT BAKGRUNDER___"
    #if is_background(tile_data):
        #return None
    
    return tile_data

#Worker function (runs in each thread) 
def process_tile(x_pos, y_pos, tile_len_x, tile_len_y):
    """
    Read tile, check background, optionally save, return result dict.
    Returns None for background or failed tiles.
    """
    try:
        tile_data = read_tile(x_pos, y_pos, tile_len_x, tile_len_y)
        if tile_data is None:
            return None  #if background tile, skip

        if SAVE_TO_DISK:
            OUTPUT_DIR.mkdir(exist_ok=True)
            Image.fromarray(tile_data).save(OUTPUT_DIR / f"tile_{y_pos:06d}_{x_pos:06d}.jpg", quality=90)
        return {"x": x_pos, "y": y_pos, "tile": tile_data} #position + image

    except Exception as e:
        print(f"  Tile ({x_pos},{y_pos}) failed: {e}")
        return None

def run_pipeline():
    """
    Tile the whole image using a thread pool.
    Returns a list of dicts: [{"x", "y", "tile"}, ...]
    """
    #Open once just to read image dimensions/size
    with rasterio.open(FILE_PATH) as ds:
        image_width, image_height = ds.width, ds.height

    coords = list(iter_tile_coords(image_width, image_height))
    total  = len(coords)
    print(f"Image   : {image_width} x {image_height} px")
    print(f"Tiles   : {total}  ({TILE_SIZE_X}x{TILE_SIZE_Y}px, {OVERLAP}px overlap)")
    print(f"Workers : {MAX_WORKERS}")
    
    "___INKLUDERA ENDAST DENNA KOD OM DU SKA TA BORT BAKGRUNDER___"
    #print(f"Tissue  : keeping tiles with at least {MIN_TISSUE_FRAC*100:.0f}% tissue\n")

    results = []
    skipped = 0
    done    = 0
    #Thread pool to process tiles in parallel with a memory brake
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool: 
        futures = []
        for col, row, w, h in coords:
            futures.append(pool.submit(process_tile, col, row, w, h)) #each tile becomes a task
            #Memory brake: drain batch before queuing more
            #Prevents thousands of tile arrays accumulating in RAM
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
                    print(f"  Progress: {done}/{total} | skipped {skipped}")

        #Drain any remaining futures after the loop
        for fut in as_completed(futures): #add valid tiles, count skipped ones
            res = fut.result()
            done += 1
            if res is None:
                skipped += 1
            else:
                results.append(res)

    kept = total - skipped
    print(f"\nTotal tiles   : {total}")
    print(f"Background    : {skipped}  ({skipped/total*100:.1f}%)")
    print(f"Tissue tiles  : {kept}  ({kept/total*100:.1f}%)")
    print(f"Ready to send : {len(results)} tiles")
    return results

tiles = run_pipeline()