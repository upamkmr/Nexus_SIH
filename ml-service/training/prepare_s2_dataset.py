import os
import argparse
import rasterio
from rasterio.windows import Window
import numpy as np
import xml.etree.ElementTree as ET
import glob

def find_baseline_in_safe(safe_dir):
    xml_files = glob.glob(os.path.join(safe_dir, "MTD_MSIL2A.xml"))
    if not xml_files:
        return None
    try:
        tree = ET.parse(xml_files[0])
        root = tree.getroot()
        # Search for PROCESSING_BASELINE
        for elem in root.iter():
            if elem.tag.endswith('PROCESSING_BASELINE'):
                return float(elem.text.strip())
    except Exception:
        pass
    return None

def prepare_s2_tile(b04_path, b03_path, b02_path, b08_path, out_path, window=None, baseline=None):
    """
    Reads Sentinel-2 B04, B03, B02, B08 (10m) bands (.jp2 or .tif)
    Stacks them in the canonical order and saves a single 4-band uint16 GeoTIFF.
    """
    bands = [b04_path, b03_path, b02_path, b08_path]
    for b in bands:
        if not os.path.exists(b):
            raise FileNotFoundError(f"Input band file not found: {b}")

    # Detect SAFE dir
    safe_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(b04_path)))))
    if baseline is None and safe_dir.endswith('.SAFE'):
        baseline = find_baseline_in_safe(safe_dir)

    # Open first band to get metadata
    with rasterio.open(b04_path) as src:
        meta = src.meta.copy()
        base_crs = src.crs
        base_transform = src.transform
        base_width = src.width
        base_height = src.height

    # Validate remaining bands against the base metadata
    for b in [b03_path, b02_path, b08_path]:
        with rasterio.open(b) as b_src:
            if b_src.width != base_width or b_src.height != base_height:
                raise ValueError(f"Shape mismatch! {b} has shape ({b_src.width}, {b_src.height}) but B04 has ({base_width}, {base_height})")
            if b_src.crs != base_crs:
                raise ValueError(f"CRS mismatch! {b} has CRS {b_src.crs} but B04 has {base_crs}")
            if b_src.transform != base_transform:
                raise ValueError(f"Transform mismatch! {b} transform does not match B04")

    meta.update({
        "count": 4,
        "driver": "GTiff",
        "dtype": "uint16"
    })
    
    if window is not None:
        win = Window(*window)
        meta["transform"] = rasterio.windows.transform(win, src.transform)
        meta["width"] = win.width
        meta["height"] = win.height
    else:
        win = None
            
    print(f"Base Profile: {meta}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    
    baseline_val = float(baseline) if baseline else 0.0
    boa_offset = 1000 if baseline_val >= 4.0 else 0

    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.update_tags(PROCESSING_BASELINE=str(baseline_val), BOA_ADD_OFFSET=str(boa_offset))
        for i, b in enumerate(bands):
            with rasterio.open(b) as b_src:
                print(f"Reading {b} ...")
                if win is not None:
                    arr = b_src.read(1, window=win)
                else:
                    arr = b_src.read(1)
                
                # Sentinel-2 data is strictly positive DNs (uint16)
                dst.write(arr.astype(np.uint16), i + 1)
                
    print(f"Successfully wrote stacked 4-band GeoTIFF to: {out_path}")
    print(f"Output Metadata: Shape=({meta['height']}, {meta['width']}), dtype={meta['dtype']}, CRS={meta.get('crs')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Sentinel-2 L2A datasets by stacking 10m bands")
    parser.add_argument("--b04", required=True, help="Path to B04 (Red) file (.jp2 or .tif)")
    parser.add_argument("--b03", required=True, help="Path to B03 (Green) file")
    parser.add_argument("--b02", required=True, help="Path to B02 (Blue) file")
    parser.add_argument("--b08", required=True, help="Path to B08 (NIR) file")
    parser.add_argument("--out", default="../../data/raw/sentinel2/stacked_tile.tif", help="Output GeoTIFF path")
    parser.add_argument("--window", type=int, nargs=4, metavar=('X_OFF', 'Y_OFF', 'WIDTH', 'HEIGHT'), 
                        help="Optional window to crop (pixels): col_off row_off width height")
    parser.add_argument("--baseline", type=float, help="Force Processing Baseline (e.g. 4.00)")

    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.normpath(os.path.join(base_dir, args.out))
    
    prepare_s2_tile(args.b04, args.b03, args.b02, args.b08, out_path, window=args.window, baseline=args.baseline)
