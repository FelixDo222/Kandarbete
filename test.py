from API_client.client import api
from API_client.common.models import SlideInput, ScanInput, ImageInput
from uuid import uuid4
from datetime import datetime, UTC
import cv2

image_path = "tiles/tile_000000_000000.jpg"
img = open(image_path, "rb").read()
date = datetime.now(UTC)

slide =  SlideInput(
            code="Hello", 
            study_id=78,
            slide_suffix="",   
            notes="Test", 
            sample_type_id= 9)

with api.create_session() as session:
    # Replace `function` with the name of the specific API function you wish to call
    response = api.create_slide(session, slide)
    print(response)
    scan = ScanInput(
        uuid = str(uuid4()),
        slide_id = response.id,
        device_id= 1,
        scan_date = date,
        notes = "Test scan"
    )
    response = api.create_scan(session, scan)
    image = ImageInput(
        file_name = "tile_000000_000000.jpg",
        scan_id = response.id,
        uuid= uuid4(),
        notes = "Test image",
        scan_date= date,
        grid_x = 0,
        grid_y = 0,
        focus_value = 1,
        focus_height = 0.0,
        pos_x = 0.0,
        pos_y = 0.0,
        pos_z = 0.0
    )
    response = api.upload_image_and_metadata(session, image, img)

