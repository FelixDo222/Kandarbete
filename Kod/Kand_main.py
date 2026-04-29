import os
import threading #runs things in parallell
import numpy as np #handles arrays, used for image processing
import rasterio #reads big .tif images efficiently
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2

FILE_PATH       = "data/file_example_TIFF_10MB.tiff"
TILE_SIZE_X     = 1280
TILE_SIZE_Y     = 720
OVERLAP         = 64       
MAX_WORKERS     = 6 #Based on CPU cores and RAM, adjust if needed.
JPG_QUALITY     = 95 #1-100, higher is better quality but larger file size
SAVE_TO_DISK    = True         
OUTPUT_DIR      = Path("tiles")
MAX_IN_FLIGHT   = MAX_WORKERS * 3   #cap queued tasks to control RAM
thread_local    = threading.local() #each thread gets its own rasterio handle, avoids thread-safety issues

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
    for y_pos in range(0, image_height, step_y):
        for x_pos in range(0, image_width, step_x):
            tile_len_x = min(TILE_SIZE_X, image_width  - x_pos)
            tile_len_y = min(TILE_SIZE_Y, image_height - y_pos)
            yield x_pos, y_pos, tile_len_x, tile_len_y


def read_tile(x_pos, y_pos, tile_len_x, tile_len_y): #(x,y,width,height)
    """
    Read one tile using rasterio's window and return a uint8 numpy array (H, W, C).
    Returns None if the tile is background.
    """
    ds   = get_dataset()  #dataset, image but not loaded into RAM
    y_pos_bottom = ds.height - y_pos  - tile_len_y
    tile_data_raw = ds.read(window=rasterio.windows.Window(x_pos, y_pos_bottom, tile_len_x, tile_len_y))   #only loads a small part of the huge image, data.shape = (channels, height, width), (3, 512, 512)
    tile_data = np.moveaxis(tile_data_raw, 0, -1)  #moves channels from position 0 to the end,converts from (C, H, W) → (H, W, C)

    #Normalise to uint8 if needed
    RGB_scaler = 255
    if tile_data.dtype != np.uint8:   #only uint8/uint16 expected, if not crash early. uint8 already in 0-255, pass through
        low, high = tile_data.min(), tile_data.max()
        tile_data = ((tile_data - low) / (high - low + 1e-9) * RGB_scaler).astype(np.uint8)
    
    #Converts RBGA to RGB by dropping alpha(A), if present. JPG dosen't support A
    if tile_data.shape[2] == 4:
        tile_data = np.delete(tile_data, 3, axis=2)
    return tile_data
 
#Worker function (runs in each thread) 
def process_tile(x_pos, y_pos, tile_len_x, tile_len_y, JPG_QUALITY):
    """
    Read tile, check background, optionally save, return result dict.
    Returns None for background or failed tiles.
    """
    try:
        tile_data = read_tile(x_pos, y_pos, tile_len_x, tile_len_y)
        if SAVE_TO_DISK:
            if TILE_SIZE_X == tile_len_x and TILE_SIZE_Y == tile_len_y:
                tile_data = cv2.cvtColor(tile_data, cv2.COLOR_RGB2BGR) #cv2 expects BGR, not RGB therefore convert before saving
                OUTPUT_DIR.mkdir(exist_ok=True)
                cv2.imwrite(str(OUTPUT_DIR / f"tile_Y{y_pos:06d}_X{x_pos:06d}.jpg"), tile_data,[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])
            else:
                height, width, channel = tile_data.shape
                right_padding = TILE_SIZE_X - tile_len_x
                top_padding = TILE_SIZE_Y - tile_len_y
                padded_tile = np.full((height + top_padding, width + right_padding, channel), 255, dtype=tile_data.dtype)
                padded_tile[top_padding:, :width, :] = tile_data
                padded = cv2.cvtColor(padded_tile, cv2.COLOR_RGB2BGR)
                OUTPUT_DIR.mkdir(exist_ok=True)
                cv2.imwrite(str(OUTPUT_DIR / f"tile_Y{y_pos:06d}_X{x_pos:06d}.jpg"), padded,[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])
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

    coords = list(iter_tile_coords(image_width, image_height)) #plans cuts, generates every tile position (x,y,w,h) across the image
    total  = len(coords) #num. of tiles image was divided into
    print(f"Image   : {image_width} x {image_height} px")
    print(f"Tiles   : {total}  ({TILE_SIZE_X}x{TILE_SIZE_Y}px, {OVERLAP}px overlap)")
    print(f"Workers : {MAX_WORKERS}")
    
    results = [] #successfully processd tiles in dict form {x: tile position in original image (left edge), y: tile position in original image (top edge), tile: tile data as numpy array}
    skipped = 0 #tiles the failed 
    done    = 0 #finished tiles (success + failed)
    #Thread pool to process tiles in parallel with a memory brake
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool: #6 independent worker threads
        futures = [] #"promises" for future results of worker tasks while workers are busy. Holds up to 18 tasks before the brake hits (drains memory before more are added)
        for x_pos, y_pos, tile_len_x, tile_len_y in coords: #loops thrpugh every tile position
            futures.append(pool.submit(process_tile, x_pos, y_pos, tile_len_x, tile_len_y)) #submit tile coords to worker, each becomes a task. Worker gives image + position
            
            #Memory brake: drain batch before queuing more, prevents thousands of tile arrays accumulating in RAM
            if len(futures) >= MAX_IN_FLIGHT: #if 18 tasks are waiting then no more tasks are submitted until one/more are done. Less than 18? Drain.
                for fut in as_completed(futures): #waits for worker to finish
                    res = fut.result() #reutrn value from process_tile()
                    done += 1
                    if res is None: #if process_tile() returned None then tile either failed or was blank
                        skipped += 1
                    else:
                        results.append(res)
                futures = [] #once all tasks in batch are done, futures are reset so next batch can be submitted

                if done % 1000 == 0:
                    print(f"  Progress: {done}/{total} | skipped {skipped}")

        #Drain: drains any remaining futures after the loop (i.e if there's less than 18)
        for fut in as_completed(futures): #same as above
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

"""""___Background Filter___"

MIN_TISSUE_FRAC = 0.05 #keep tiles with at least this fraction of content 
BG_WHITE        = 230  #pixels above this = white background
BG_BLACK        = 10   #pixels below this = black background

def is_background(tile_data):
    
    Return True if the tile is mostly empty (white or black background).
    arr shape: (H, W, 3) uint8
    - White background: pixels > BG_WHITE
    - Black background: pixels < BG_BLACK
    A tile needs MIN_TISSUE_FRAC of pixels that are NEITHER white nor black.

    gray        = tile_data.mean(axis=2)          #converts RGB to grayscale
    not_white   = np.mean(gray < BG_WHITE)  #fraction that is not white/bg
    not_black   = np.mean(gray > BG_BLACK)  #fraction that is not black/bg
    tissue_frac = min(not_white, not_black) #must pass both checks
    return tissue_frac < MIN_TISSUE_FRAC #if less than 5% real content → skip tile
    
    ___In read_tile()___
    if is_background(tile_data):
        return None
    
    ___In process_tile()___       
    if tile_data is None:
        return None  #if background tile, skip
    
    ___In run_pipeline()___
    print(f"Tissue  : keeping tiles with at least {MIN_TISSUE_FRAC*100:.0f}% tissue\n")
    
    """
