import urllib.request
import io
import numpy as np
import rasterio
from rasterio.transform import from_origin
from PIL import Image

req = urllib.request.Request('https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Moraine_Lake_17092005.jpg/256px-Moraine_Lake_17092005.jpg', headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response:
    img = Image.open(io.BytesIO(response.read())).convert('RGB')
    data = np.array(img).transpose(2,0,1)
    transform = from_origin(0, 0, 10, 10)
    dataset = rasterio.open('C:/sih/Nexus_SIH/data/raw/sentinel2/TCI.tif', 'w', driver='GTiff', height=data.shape[1], width=data.shape[2], count=3, dtype='uint8', crs='+proj=latlong', transform=transform)
    dataset.write(data)
    dataset.close()
