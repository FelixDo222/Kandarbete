from PIL import Image

# Open TIFF image
img = Image.open("data/file_example_TIFF_10MB.tiff")

right_border = 50

width, height = img.size

# Create new image: only width increases
new_img = Image.new("RGB", (width + right_border, height), "black")

# Paste original image on the left
new_img.paste(img, (0, 0))

new_img.save("output.tif")