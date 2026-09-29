import rasterio
from rasterio.windows import Window
import requests

url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/31/U/DQ/2023/10/S2A_31UDQ_20231011_0_L2A/TCI.tif"

print('Downloading...', url)
try:
    with rasterio.open(url) as src:
        # Just grab a 1024x1024 window to keep the file size manageable for testing
        window = Window(1000, 1000, 1024, 1024)
        data = src.read(window=window)
        meta = src.meta.copy()
        meta.update({
            "height": 1024,
            "width": 1024,
            "transform": src.window_transform(window)
        })
        
        out_path = r"C:\sih\Nexus_SIH\data\raw\sentinel2\TCI.tif"
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data)
            # Add BOA tags just in case
            dst.update_tags(PROCESSING_BASELINE="05.09", BOA_ADD_OFFSET="1000")
            
    print('Saved to', out_path)
except Exception as e:
    print('Error:', e)
