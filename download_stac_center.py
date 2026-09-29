import requests
import rasterio
from rasterio.windows import Window
import os
import numpy as np

res = requests.post("https://earth-search.aws.element84.com/v1/search", json={
    "collections": ["sentinel-2-l2a"],
    "limit": 5
}, headers={"User-Agent": "Mozilla/5.0"})
data = res.json()

out_path = r'C:\sih\Nexus_SIH\data\raw\sentinel2\TCI.tif'

for item in data['features']:
    tci_url = item['assets']['visual']['href']
    print('Trying', tci_url)
    try:
        with rasterio.open(tci_url) as src:
            # Grab from center
            w, h = src.width, src.height
            window = Window(w//2 - 512, h//2 - 512, 1024, 1024)
            img_data = src.read(window=window)
            if np.mean(img_data) > 10:  # Not completely black
                meta = src.meta.copy()
                meta.update({
                    "height": 1024,
                    "width": 1024,
                    "transform": src.window_transform(window)
                })
                with rasterio.open(out_path, "w", **meta) as dst:
                    dst.write(img_data)
                    dst.update_tags(PROCESSING_BASELINE="05.09", BOA_ADD_OFFSET="1000")
                print("Success! Mean:", np.mean(img_data))
                break
    except Exception as e:
        print('Error:', e)
