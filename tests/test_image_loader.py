import os
from src.image_loader import load_images_from_folder

def test_load_images_from_folder(tmp_path):
    # Create a temporary directory with some images
    image_folder = tmp_path / "images"
    image_folder.mkdir()
    image_path = image_folder / "test_image.png"
    image_path.write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xdc\xccY\xe7\x00\x00\x00\x00IEND\xaeB`\x82')

    images = load_images_from_folder(image_folder)
    assert len(images) == 1, "There should be one image loaded"
    assert images[0].shape == (1, 3, 224, 224), "Image shape should be (1, 3, 224, 224)"
