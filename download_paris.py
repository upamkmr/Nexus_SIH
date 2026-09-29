import requests
import rasterio
from rasterio.windows import Window
import numpy as np

res = requests.post("https://earth-search.aws.element84.com/v1/search", json={
    "collections": ["sentinel-2-l2a"],
    "intersects": {
        "type": "Point",
        "coordinates": [2.35, 48.85] # Paris
    },
    "limit": 1
}, headers={"User-Agent": "Mozilla/5.0"})
data = res.json()
item = data['features'][0]
tci_url = item['assets']['visual']['href']
print("Found Paris tile:", tci_url)

out_path = r'C:\sih\Nexus_SIH\data\raw\sentinel2\TCI.tif'
with rasterio.open(tci_url) as src:
    window = Window(5000, 5000, 1024, 1024)
    img_data = src.read(window=window)
    print("Mean:", np.mean(img_data))
    meta = src.meta.copy()
    meta.update({
        "height": 1024,
        "width": 1024,
        "transform": src.window_transform(window)
    })
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(img_data)
        dst.update_tags(PROCESSING_BASELINE="05.09", BOA_ADD_OFFSET="1000")
print("Saved Paris tile successfully!")
