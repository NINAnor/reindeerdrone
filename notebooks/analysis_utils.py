import geopandas as gpd
import pathlib
import json
from shapely.geometry import box, Polygon
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import seaborn as sns
import os

def parse_json(json_path):
    # Load your JSON file
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    for item in data:
        image_id = item.get("file_upload").split("-")[1].split(".")[0]

        # Get image dimensions if available (to convert normalized coordinates to pixels)
        original_width, original_height = None, None
        if item.get("annotations"):
            first_res = item["annotations"][0].get("result", [])[0]
            original_width = first_res.get("original_width")
            original_height = first_res.get("original_height")
        
        for ann in item.get("annotations", []):
            for res in ann.get("result", []):
                res_type = res.get("type")
                value = res.get("value", {})
                
                # Determine annotation type and build geometry.
                if res_type == "rectanglelabels":
                    annotation_type = "bbox"
                    label = value.get("rectanglelabels", [None])[0]
                    x = value.get("x")
                    y = value.get("y")
                    width = value.get("width")
                    height = value.get("height")
                    # Convert normalized values (if in percentages) to pixels if dimensions are provided.
                    if original_width and original_height:
                        x = x * original_width / 100
                        y = y * original_height / 100
                        width = width * original_width / 100
                        height = height * original_height / 100
                    geom = box(x, y, x + width, y + height)
                elif res_type == "polygonlabels":
                    annotation_type = "polygon"
                    label = value.get("polygonlabels", [None])[0]
                    points = value.get("points")
                    # Convert points from normalized to pixels if possible
                    if original_width and original_height:
                        points = [
                            (pt[0] * original_width / 100, pt[1] * original_height / 100)
                            for pt in points
                        ]
                    geom = Polygon(points)
                else:
                    continue
                
                rows.append({
                    "ImageID": image_id,
                    "LabelType": label,
                    "annotation_type": annotation_type,
                    "geometry": geom
                })

    gdf = gpd.GeoDataFrame(rows)
    return gdf

def plot_annotations(image_path, gdf, label_col="LabelType"):

    img = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, origin="upper")

    gdf = gdf[gdf["ImageID"] == image_path.stem]
    
    for idx, row in gdf.iterrows():
        geom = row["geometry"]
        label = row[label_col]
        if geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            x, y = geom.exterior.xy
            ax.plot(x, y, color="red", linewidth=2)
            centroid = geom.centroid
            ax.text(centroid.x, centroid.y, label, fontsize=1, color="blue",
                    ha="center", va="center", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
        elif geom.geom_type == "MultiPolygon":
            for poly in geom:
                x, y = poly.exterior.xy
                ax.plot(x, y, color="red", linewidth=2)
                centroid = poly.centroid
                ax.text(centroid.x, centroid.y, label, fontsize=12, color="blue",
                        ha="center", va="center", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    
    ax.set_title("Image with Annotations")
    plt.axis("off")
    plt.show()


def count_in_polygons(gdf):

    gdf["geometry"] = gdf.apply(
        lambda row: row.geometry.centroid if row["annotation_type"] == "bbox" else row.geometry,
        axis=1
    )
    
    points = gdf[gdf['annotation_type'] == 'bbox'].copy()
    polygons = gdf[gdf['annotation_type'] == 'polygon'].copy()
    result_list = []

    for image in gdf['ImageID'].unique():
        pts = points[points['ImageID'] == image].copy()
        polys = polygons[polygons['ImageID'] == image].copy()
        
        if polys.empty:
            pts['polygon_label'] = "open landscape"
            result_list.append(pts)
        else:
            joined = gpd.sjoin(pts, polys[['LabelType', 'geometry']], how="left", predicate="within")
            joined['polygon_label'] = joined['LabelType_right'].fillna("open landscape")
            result_list.append(joined)


    joined_points = pd.concat(result_list, ignore_index=True)

    counts = (
        joined_points.groupby(['ImageID', 'LabelType_left','polygon_label'])
        .size()
        .reset_index(name='count')
    )

    return counts


def parse_prediction_json(file_path):
    base = os.path.basename(file_path)
    if base.endswith('_pred.json'):
        image_id = base.replace('_pred.json', '')
    else:
        image_id = os.path.splitext(base)[0]
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    rows = []
    for pred in data:
        bbox = pred.get("bbox", None)
        if not bbox:
            continue

        xmin, ymin, xmax, ymax = bbox
        geom = box(xmin, ymin, xmax, ymax)
        
        score = pred.get("score", None)
        category_id = pred.get("category_id", None)

        label = "Adult" if category_id == 0 else "Calf" if category_id == 1 else None
        
        rows.append({
            "ImageID": image_id,
            "LabelType": label,
            "score": score,
            "geometry": geom
        })
    
    gdf = gpd.GeoDataFrame(rows, crs=None)
    return gdf


def assign_majority_environment(reindeer_gdf, landscape_gdf):
    
    assigned_list = []

    for image_id in reindeer_gdf['ImageID'].unique():
        reindeer_img = reindeer_gdf[reindeer_gdf['ImageID'] == image_id].copy()
        landscape_img = landscape_gdf[landscape_gdf['ImageID'] == image_id].copy()

        assigned_landtypes = []
        for idx, row in reindeer_img.iterrows():
            bbox = row.geometry
            candidate_landscapes = landscape_img[landscape_img.intersects(bbox)]

            if candidate_landscapes.empty:
                assigned_landtypes.append("Open Landscape")
                continue

            candidate_landscapes = candidate_landscapes.copy()  
            candidate_landscapes['intersection_area'] = candidate_landscapes.geometry.apply(lambda poly: bbox.intersection(poly).area)
            
            bbox_area = bbox.area
            candidate_landscapes['overlap_pct'] = candidate_landscapes['intersection_area'] / bbox_area
            
            best_candidate = candidate_landscapes.sort_values('intersection_area', ascending=False).iloc[0]
            assigned_landtypes.append(best_candidate['landtype'])
        
        reindeer_img['landtype'] = assigned_landtypes
        assigned_list.append(reindeer_img)

    return pd.concat(assigned_list, ignore_index=True)

from shapely.geometry import Polygon
from shapely.wkt import loads

def compute_iou(poly1, poly2):
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersection / union if union != 0 else 0

def evaluate_detections(detections, ground_truths, iou_threshold=0.5):
    tp = []
    fp = []
    fn = ground_truths.copy()

    # Iterate over each detection row
    for det_idx, det in detections.iterrows():
        det_poly = det['geometry'] 
        best_match = None
        best_match_idx = None
        best_iou = 0

        for gt_idx, gt in ground_truths.iterrows():
            gt_poly = gt['geometry']
            iou = compute_iou(det_poly, gt_poly)
            # Check if IoU is above threshold and labels match
            if iou >= iou_threshold and det['LabelType'] == gt['LabelType'] and iou > best_iou:
                best_iou = iou
                best_match = gt
                best_match_idx = gt_idx

        if best_match is not None:
            tp.append(det)
            # Remove the matched ground truth using its index
            if best_match_idx in fn.index:
                fn = fn.drop(best_match_idx)
        else:
            fp.append(det)

    return pd.DataFrame(tp), pd.DataFrame(fp), pd.DataFrame(fn)