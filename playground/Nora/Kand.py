import os
import threading #runs things in parallell
import numpy as np #handles arrays, used for image processing
import rasterio #reads big .tif images efficiently
from rasterio.windows import Window
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

FILE_PATH       = "25FU1231.tif"
TILE_SIZE       = 512          
OVERLAP         = 64           
MAX_WORKERS     = min(6, os.cpu_count()) #parallell threads based on CPU cores
MIN_TISSUE_FRAC = 0.05 #keep tiles with at least this fraction of content 
BG_WHITE        = 230  #pixels above this = white background
BG_BLACK        = 10   #pixels below this = black background
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
    step = TILE_SIZE - OVERLAP #512-64 = 448 pixels/step 
    seen = set() #ensures no repeated tiles

    for row in range(0, height, step): #
        for col in range(0, width, step):
            c = min(col, max(0, width  - TILE_SIZE)) #prevents tiles from going outside the image
            r = min(row, max(0, height - TILE_SIZE))

            if (c, r) in seen:
                continue
            seen.add((c, r))

            #Actual tile size (edge tiles may be smaller)
            w = min(TILE_SIZE, width  - c)
            h = min(TILE_SIZE, height - r)
            yield c, r, w, h #(x, y, width, height)

#Background filter 
def is_background(arr):
    """
    Return True if the tile is mostly empty (white or black background).
    arr shape: (H, W, 3) uint8
    - White background: pixels > BG_WHITE
    - Black background: pixels < BG_BLACK
    A tile needs MIN_TISSUE_FRAC of pixels that are NEITHER white nor black.
    """
    gray        = arr.mean(axis=2)          #converts RGB to grayscale
    not_white   = np.mean(gray < BG_WHITE)  #fraction that is not white/bg
    not_black   = np.mean(gray > BG_BLACK)  #fraction that is not black/bg
    tissue_frac = min(not_white, not_black) #must pass both checks
    return tissue_frac < MIN_TISSUE_FRAC #if less than 5% real content → skip tile

def read_tile(col, row, w, h): #(x,y,width, height)
    """
    Read one tile using rasterio's window and return a uint8 numpy array (H, W, C).
    Returns None if the tile is background.
    """
    ds   = get_ds()  #dataset, image but not loaded into RAM
    data = ds.read(window=Window(col, row, w, h))   #only loads a small part of the huge image, data.shape = (channels, height, width), (3, 512, 512)
    arr  = np.moveaxis(data, 0, -1)  #moves channels from position 0 to the end,converts from (C, H, W) → (H, W, C)

    #Normalise to uint8 if needed
    if arr.dtype == np.uint16:
        arr = (arr / 257).astype(np.uint8)  #scale 0-65535 to 0-255
    elif arr.dtype != np.uint8:   #if other -> scale to 0-255 
        lo, hi = arr.min(), arr.max()
        arr = ((arr - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)

    if is_background(arr):
        return None

    return arr

#Worker function (runs in each thread) 
def process_tile(col, row, w, h):
    """
    Read tile, check background, optionally save, return result dict.
    Returns None for background or failed tiles.
    """
    try:
        arr = read_tile(col, row, w, h)
        if arr is None:
            return None  #if background tile, skip

        if SAVE_TO_DISK:
            OUTPUT_DIR.mkdir(exist_ok=True)
            from PIL import Image
            Image.fromarray(arr).save(
                OUTPUT_DIR / f"tile_{row:06d}_{col:06d}.png",
                compress_level=1   #fast lossless PNG
            )

        return {"col": col, "row": row, "tile": arr} #position + image

    except Exception as e:
        print(f"  Tile ({col},{row}) failed: {e}")
        return None

def run_pipeline():
    """
    Tile the whole image using a thread pool.
    Returns a list of dicts: [{"col", "row", "tile"}, ...]
    """
    #Open once just to read image dimensions/size
    with rasterio.open(FILE_PATH) as ds:
        width, height = ds.width, ds.height

    coords = list(iter_tile_coords(width, height))
    total  = len(coords)
    print(f"Image   : {width} x {height} px")
    print(f"Tiles   : {total}  ({TILE_SIZE}px, {OVERLAP}px overlap)")
    print(f"Workers : {MAX_WORKERS}")
    print(f"Tissue  : keeping tiles with at least {MIN_TISSUE_FRAC*100:.0f}% tissue\n")

    results = []
    skipped = 0
    done    = 0

    #Thread pool to process tiles in parallel with a memory brake
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool: 
        futures = []
        for col, row, w, h in coords:
            futures.append(pool.submit(process_tile, col, row, w, h)) #each tile becomes a task

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
print("hello world2")