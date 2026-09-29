import rasterio
import numpy as np

try:
    with rasterio.open(r'C:\sih\Nexus_SIH\data\raw\sentinel2\TCI.tif') as src:
        data = src.read()
        print('Shape:', data.shape)
        print('Dtype:', data.dtype)
        print('Min:', np.min(data))
        print('Max:', np.max(data))
        print('Mean:', np.mean(data))
        unique_vals = np.unique(data)
        print('Unique values:', len(unique_vals), unique_vals[:10])
except Exception as e:
    print('Error:', e)
