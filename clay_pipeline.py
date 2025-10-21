"""
clay_pipeline.py

FastAPI app that loads the Clay MAE model from a checkpoint, queries the Sentinel-2 STAC API
for images for a given latitude/longitude and date range, stacks them into a datacube,
normalizes using metadata mean/std, computes embeddings (class token),
and returns JSON with:
 - times: ISO datetimes of acquisitions
 - thumbnails: base64 PNG previews
 - embeddings: list of float embeddings
"""

import os
import io
import math
import base64
import datetime
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from torchvision.transforms import v2
from box import Box

import pystac_client
import stackstac
import yaml
import geopandas as gpd
import pandas as pd
from shapely import Point
from rasterio.enums import Resampling

try:
    from claymodel.module import ClayMAEModule
except Exception as e:
    ClayMAEModule = None

# ------------------- Configuration -------------------
STAC_API = os.environ.get("STAC_API", "https://earth-search.aws.element84.com/v1")
COLLECTION = "sentinel-2-l2a"
MODEL_CKPT = os.environ.get("MODEL_CKPT", "/data/clay-v1.5.ckpt")
METADATA_PATH = os.environ.get("METADATA_PATH", "/data/metadata.yaml")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

OUT_SHAPE = (256, 256)

# ------------------- FastAPI -------------------
app = FastAPI(title="Clay pipeline API")

class ImageRequest(BaseModel):
    lat: float
    lon: float
    start_date: str
    end_date: str


_model = None
_transform = None
_metadata = None

# ------------------- Helpers -------------------
def load_metadata(meta_path: str):
    """Load metadata from yaml file."""
    global _metadata, _transform
    with open(meta_path, "r") as f:
        _metadata = Box(yaml.safe_load(f))
    
    platform = COLLECTION
    mean = []
    std = []
    
    # Use the band names to get the correct values in the correct order
    band_names = ["blue", "green", "red", "nir"]
    for band in band_names:
        mean.append(_metadata[platform].bands.mean[str(band)])
        std.append(_metadata[platform].bands.std[str(band)])
    
    # Prepare the normalization transform function
    _transform = v2.Compose([
        v2.Normalize(mean=mean, std=std),
    ])
    print(f"Loaded metadata for {len(band_names)} bands.")


def normalize_timestamp(date):
    """Convert timestamp to sin/cos encoding (week + hour)."""
    week = date.isocalendar().week * 2 * np.pi / 52
    hour = date.hour * 2 * np.pi / 24
    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))


def normalize_latlon(lat, lon):
    """Convert lat/lon to sin/cos encoding."""
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))


def tensor_to_thumbnail(raw_data, max_side=256):
    """Convert raw satellite data to base64 PNG thumbnail."""
    if isinstance(raw_data, torch.Tensor):
        raw_data = raw_data.cpu().numpy()
    
    # Use RGB bands for thumbnail (blue, green, red)
    if raw_data.shape[0] >= 3:
        rgb = raw_data[:3]  # blue, green, red
    else:
        rgb = np.vstack([raw_data[0:1]] * 3)

    rgb = np.transpose(rgb, (1, 2, 0))
    
    # Use the same approach as the notebook: display raw satellite data
    # Scale to 0-255 range using typical satellite data range (0-2000)
    rgb = np.clip(rgb / 2000.0 * 255.0, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(rgb)
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def query_stac(lat: float, lon: float, start_date: str, end_date: str, limit: int = 100):
    """Query STAC API for Sentinel-2 images."""
    catalog = pystac_client.Client.open(STAC_API)
    search = catalog.search(
        collections=[COLLECTION],
        datetime=f"{start_date}/{end_date}",
        bbox=(lon - 1e-5, lat - 1e-5, lon + 1e-5, lat + 1e-5),
        max_items=limit,
        query={"eo:cloud_cover": {"lt": 80}},
    )
    
    all_items = search.item_collection()
    
    # Reduce to one per date (remove duplicates)
    items = []
    dates = []
    for item in all_items:
        if item.datetime.date() not in dates:
            items.append(item)
            dates.append(item.datetime.date())
    
    return items


def items_to_datacube(items, lat: float, lon: float):
    """Convert STAC items to datacube using stackstac."""
    if not items:
        return None
    
    # Extract coordinate system from first item
    epsg = items[0].properties["proj:code"]
    
    # Convert point of interest into the image projection
    poidf = gpd.GeoDataFrame(
        pd.DataFrame(),
        crs="EPSG:4326",
        geometry=[Point(lon, lat)],
    ).to_crs(epsg)

    coords = poidf.iloc[0].geometry.coords[0]
    
    # Create bounds in projection
    size = 256
    gsd = 10
    bounds = (
        coords[0] - (size * gsd) // 2,
        coords[1] - (size * gsd) // 2,
        coords[0] + (size * gsd) // 2,
        coords[1] + (size * gsd) // 2,
    )
    
    # Retrieve the pixel values
    epsg_code = items[0].properties["proj:code"].split(":")[-1]
    stack = stackstac.stack(
        items,
        bounds=bounds,
        snap_bounds=False,
        epsg=int(epsg_code),
        resolution=gsd,
        dtype="float64",
        rescale=False,
        fill_value=np.float64(0.0),
        assets=["blue", "green", "red", "nir"],
        resampling=Resampling.nearest,
    )
    
    return stack.compute()


def load_model(ckpt_path=MODEL_CKPT, metadata_path=METADATA_PATH):
    """Load Clay model from checkpoint."""
    global _model
    if ClayMAEModule is None:
        raise RuntimeError("ClayMAEModule not importable. Install clay-model properly.")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    torch.set_default_device(DEVICE)
    _model = ClayMAEModule.load_from_checkpoint(
        ckpt_path,
        model_size="large",
        metadata_path=metadata_path,
        dolls=[16, 32, 64, 128, 256, 768, 1024],
        doll_weights=[1, 1, 1, 1, 1, 1, 1],
        mask_ratio=0.0,
        shuffle=False,
    )
    _model.eval().to(DEVICE)
    return _model


# ------------------- Startup -------------------
@app.on_event("startup")
async def startup_event():
    try:
        load_model()
        load_metadata(METADATA_PATH)
        print("Clay model loaded successfully.")
    except Exception as e:
        print("Warning: model load failed:", e)


# ------------------- Endpoint -------------------
@app.post("/generate-embeddings")
async def generate_embeddings(req: ImageRequest):
    try:
        datetime.datetime.fromisoformat(req.start_date)
        datetime.datetime.fromisoformat(req.end_date)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid dates. Use YYYY-MM-DD format.")

    # Query STAC API
    items = query_stac(req.lat, req.lon, req.start_date, req.end_date)
    if not items:
        return {"times": [], "embeddings": [], "thumbnails": []}

    # Convert to datacube
    stack = items_to_datacube(items, req.lat, req.lon)
    if stack is None:
        return {"times": [], "embeddings": [], "thumbnails": []}

    # Load model if not loaded
    global _model
    if _model is None:
        _model = load_model()

    # Prepare datetimes embedding
    datetimes = stack.time.values.astype("datetime64[s]").tolist()
    times = [normalize_timestamp(dat) for dat in datetimes]
    week_norm = [dat[0] for dat in times]
    hour_norm = [dat[1] for dat in times]

    # Prepare lat/lon embedding
    latlons = [normalize_latlon(req.lat, req.lon)] * len(times)
    lat_norm = [dat[0] for dat in latlons]
    lon_norm = [dat[1] for dat in latlons]

    # Normalize pixels
    pixels = torch.from_numpy(stack.data.astype(np.float32))
    pixels = _transform(pixels)

    # Prepare additional information
    platform = COLLECTION
    mean = []
    std = []
    waves = []
    
    # Use the band names to get the correct values in the correct order
    for band in stack.band:
        mean.append(_metadata[platform].bands.mean[str(band.values)])
        std.append(_metadata[platform].bands.std[str(band.values)])
        waves.append(_metadata[platform].bands.wavelength[str(band.values)])

    datacube = {
        "platform": platform,
        "time": torch.tensor(
            np.hstack((week_norm, hour_norm)),
            dtype=torch.float32,
            device=DEVICE,
        ),
        "latlon": torch.tensor(
            np.hstack((lat_norm, lon_norm)), dtype=torch.float32, device=DEVICE
        ),
        "pixels": pixels.to(DEVICE),
        "gsd": torch.tensor(stack.gsd.values, device=DEVICE),
        "waves": torch.tensor(waves, device=DEVICE),
    }

    # Generate embeddings
    embeddings, raw_images, times = [], [], []
    with torch.no_grad():
        unmsk_patch, unmsk_idx, msk_idx, msk_matrix = _model.model.encoder(datacube)

        # Extract class token embeddings
        cls_embeddings = unmsk_patch[:, 0, :].cpu().numpy()

        for i in range(len(stack.time)):
            embeddings.append(cls_embeddings[i].tolist())
   
            # Pass raw satellite data instead of generating thumbnails
            raw_slice_data = stack.isel(time=i).data
            # Convert to list format for JSON serialization
            raw_images.append(raw_slice_data.tolist())
   
            # Time
            times.append(str(stack.time[i].values))

    return {"times": times, "embeddings": embeddings, "raw_images": raw_images}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
