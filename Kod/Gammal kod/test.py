import time
import pathlib
import tifffile as tiff
from concurrent.futures import ThreadPoolExecutor, as_completed


def generate_positions(length, tile_size, overlap):
    stride = tile_size - overlap
    positions = list(range(0, max(length - tile_size + 1, 1), stride))
    last_pos = max(length - tile_size, 0)
    if not positions or positions[-1] != last_pos:
        positions.append(last_pos)
    return positions


def save_tile(image, y, x, tile_h, tile_w, output_dir):
    tile = image[y:y + tile_h, x:x + tile_w]
    out_path = output_dir / f"tile_y{y}_x{x}.tif"
    tiff.imwrite(out_path, tile)


def tile_tiff(input_path, output_directory, tile_height, tile_width, overlap):
    image = tiff.imread(input_path)
    height, width = image.shape[:2]
    y_positions = generate_positions(height, tile_height, overlap)
    x_positions = generate_positions(width, tile_width, overlap)
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    tasks = [(y, x) for y in y_positions for x in x_positions]
    # IMPORTANT: limit queue pressure
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for y, x in tasks:
            futures.append(
                executor.submit(
                    save_tile,
                    image,
                    y,
                    x,
                    tile_height,
                    tile_width,
                    output_directory
                )
            )
        # drain safely (prevents memory buildup)
        for f in as_completed(futures):
            f.result()
    print(f"Done: {len(tasks)} tiles")

#Usage
start_time = time.time()
tile_tiff(input_path = "25FU1231.tif", 
          output_directory = "Example_folder", 
          tile_height = 512, 
          tile_width = 512, 
          overlap=64)
end_time = time.time()
print(f"Processing time: {end_time - start_time} seconds")