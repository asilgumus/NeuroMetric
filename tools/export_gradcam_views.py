"""Render each existing, unchanged scientific Grad-CAM slice separately."""
from pathlib import Path
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainage import sha256

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-folder", type=Path, default=ROOT / "artifacts/alzheimer/prepared/OAS1_0030_MR1")
parser.add_argument("--gradcam-folder", type=Path, default=ROOT / "artifacts/alzheimer/gradcam_OAS1_0030")
parser.add_argument("--asset-prefix", default="oasis30-gradcam")
args = parser.parse_args()
if not args.asset_prefix.replace("-", "").isalnum():
    raise ValueError("Asset prefix must contain only letters, digits and hyphens")
folder = args.gradcam_folder
metadata = json.loads((folder / "metadata.json").read_text())
source = args.input_folder / "sfcn.npy"
if sha256(source) != metadata["input_sha256"]:
    raise ValueError("Input differs from the MRI used for attribution")
brain = np.load(source, allow_pickle=False).astype(np.float32)
heat = np.load(folder / "signed_gradcam.npy", allow_pickle=False)
for name, axis, index in [("axial",2,80),("coronal",1,96),("sagittal",0,80)]:
    image = np.rot90(np.take(brain,index,axis=axis))
    overlay = np.rot90(np.take(heat,index,axis=axis))
    fig = plt.figure(figsize=(5,5),facecolor="#080e17")
    ax = fig.add_axes([0,0,1,1])
    ax.set_facecolor("#080e17")
    ax.imshow(image,cmap="gray",vmin=0,vmax=np.percentile(brain[brain>0],99))
    ax.imshow(overlay,cmap="coolwarm",vmin=-1,vmax=1,alpha=np.where(image>0,.55*np.abs(overlay),0))
    ax.axis("off")
    fig.savefig(ROOT / ("assets/"+args.asset_prefix+"-"+name+".png"),dpi=140,facecolor=fig.get_facecolor())
    plt.close(fig)
print("Exported three views from the existing attribution; no new prediction")
