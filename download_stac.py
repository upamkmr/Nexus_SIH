import requests
import rasterio
from rasterio.windows import Window
import os

print("Searching Earth Search STAC...")
headers = {"User-Agent": "Mozilla/5.0"}
res = requests.post("https://earth-search.aws.element84.com/v1/search", json={
    "collections": ["sentinel-2-l2a"],
    "limit": 1
}, headers=headers)
try:
    data = res.json()
except:
    print("Response:", res.text)
    exit(1)

if not data.get("features"):
    print("No features found")
    exit(1)

item = data["features"][0]
print("Found item:", item["id"])
tci_url = item["assets"]["visual"]["href"]
print("TCI URL:", tci_url)

out_path = r"C:\sih\Nexus_SIH\data\raw\sentinel2\TCI.tif"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

try:
    with rasterio.open(tci_url) as src:
        window = Window(2000, 2000, 1024, 1024)
        print("Reading data...")
        img_data = src.read(window=window)
        meta = src.meta.copy()
        meta.update({
            "height": 1024,
            "width": 1024,
            "transform": src.window_transform(window)
        })
        print("Writing locally...")
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(img_data)
            dst.update_tags(PROCESSING_BASELINE="05.09", BOA_ADD_OFFSET="1000")
            
    print("Saved real Sentinel-2 TCI successfully to", out_path)
except Exception as e:
    print("Error:", e)
