"""
Scribble Annotation Generator for RefHuman/UniPHD Dataset

This script automatically generates scribble prompts from segmentation polygons
following the methodology used in the Referring Human Pose and Mask Estimation paper.

Scribbles are positional prompts (NOT supervision masks) that:
- Cover ~1-5% of the instance area
- Lie strictly inside the instance mask
- Are stored as COCO-style coordinate lists

Usage:
    python generate_scribbles.py --input RefHuman_train.json --output RefHuman_train_scribbles.json
    python generate_scribbles.py --input RefHuman_val.json --output RefHuman_val_scribbles.json
"""

import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
from skimage.morphology import skeletonize, medial_axis
from skimage.measure import label, regionprops
import random
import warnings

warnings.filterwarnings('ignore')


def polygon_to_mask(segmentation: List[List[float]], height: int, width: int) -> np.ndarray:
    """
    Convert COCO segmentation polygon(s) to a binary mask.
    
    Args:
        segmentation: List of polygon coordinates [[x1,y1,x2,y2,...], ...]
        height: Image height
        width: Image width
    
    Returns:
        Binary mask of shape (height, width)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if not segmentation:
        return mask
    
    for polygon in segmentation:
        if len(polygon) < 6:  # Need at least 3 points (6 coordinates)
            continue
        
        # Reshape to (N, 1, 2) format for cv2.fillPoly
        pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
    
    return mask


def extract_skeleton(mask: np.ndarray) -> np.ndarray:
    """
    Extract skeleton/centerline from binary mask using skeletonization.
    
    Args:
        mask: Binary mask of shape (height, width)
    
    Returns:
        Binary skeleton mask
    """
    if mask.sum() == 0:
        return np.zeros_like(mask)
    
    # Use skeletonization (morphological thinning)
    skeleton = skeletonize(mask > 0)
    
    return skeleton.astype(np.uint8)


def extract_medial_axis_skeleton(mask: np.ndarray) -> np.ndarray:
    """
    Extract skeleton using medial axis transform for smoother results.
    
    Args:
        mask: Binary mask of shape (height, width)
    
    Returns:
        Binary skeleton mask
    """
    if mask.sum() == 0:
        return np.zeros_like(mask)
    
    try:
        # Medial axis transform
        skel, distance = medial_axis(mask > 0, return_distance=True)
        return skel.astype(np.uint8)
    except:
        # Fallback to regular skeletonization
        return extract_skeleton(mask)


def get_skeleton_points(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """
    Extract all skeleton pixel coordinates.
    
    Args:
        skeleton: Binary skeleton mask
    
    Returns:
        List of (x, y) coordinates
    """
    points = np.where(skeleton > 0)
    # Return as (x, y) format (column, row)
    return list(zip(points[1].tolist(), points[0].tolist()))


def subsample_skeleton_points(
    points: List[Tuple[int, int]], 
    target_coverage: float = 0.03,
    instance_area: int = 1000,
    min_points: int = 5,
    max_points: int = 100
) -> List[Tuple[int, int]]:
    """
    Subsample skeleton points to achieve sparse scribble coverage.
    
    Args:
        points: List of skeleton (x, y) coordinates
        target_coverage: Target coverage as fraction of instance area (0.01-0.05)
        instance_area: Area of the instance mask in pixels
        min_points: Minimum number of points to keep
        max_points: Maximum number of points to keep
    
    Returns:
        Subsampled list of (x, y) coordinates
    """
    if not points:
        return []
    
    # Calculate target number of points based on coverage
    # Each point contributes roughly 1 pixel; with dilation it becomes more
    target_pixels = int(instance_area * target_coverage)
    target_num_points = max(min_points, min(max_points, target_pixels // 2))
    
    # If we have fewer points than target, return all
    if len(points) <= target_num_points:
        return points
    
    # Uniform subsampling along the skeleton
    # Sort points to maintain connectivity where possible
    indices = np.linspace(0, len(points) - 1, target_num_points, dtype=int)
    subsampled = [points[i] for i in indices]
    
    return subsampled


def create_connected_polylines(
    points: List[Tuple[int, int]], 
    max_gap: int = 10
) -> List[List[Tuple[int, int]]]:
    """
    Group skeleton points into connected polyline segments.
    
    Args:
        points: List of (x, y) coordinates
        max_gap: Maximum distance between consecutive points in a polyline
    
    Returns:
        List of polylines, where each polyline is a list of (x, y) coordinates
    """
    if not points:
        return []
    
    if len(points) == 1:
        return [points]
    
    # Sort points by x then y to create a path
    sorted_points = sorted(points, key=lambda p: (p[0], p[1]))
    
    polylines = []
    current_line = [sorted_points[0]]
    
    for i in range(1, len(sorted_points)):
        prev = current_line[-1]
        curr = sorted_points[i]
        
        # Calculate distance
        dist = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
        
        if dist <= max_gap:
            current_line.append(curr)
        else:
            if len(current_line) >= 2:
                polylines.append(current_line)
            current_line = [curr]
    
    if len(current_line) >= 2:
        polylines.append(current_line)
    elif current_line and not polylines:
        polylines.append(current_line)
    
    return polylines


def dilate_scribble_mask(
    mask: np.ndarray, 
    scribble_points: List[Tuple[int, int]], 
    dilation_radius: int = 2
) -> np.ndarray:
    """
    Create a dilated scribble mask from points.
    
    Args:
        mask: Original instance mask for boundary checking
        scribble_points: List of (x, y) scribble coordinates
        dilation_radius: Radius for dilation (1-3 pixels)
    
    Returns:
        Dilated scribble mask (for visualization, not storage)
    """
    h, w = mask.shape
    scribble_mask = np.zeros((h, w), dtype=np.uint8)
    
    for x, y in scribble_points:
        if 0 <= x < w and 0 <= y < h:
            scribble_mask[y, x] = 1
    
    # Apply dilation
    if dilation_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (2*dilation_radius + 1, 2*dilation_radius + 1)
        )
        scribble_mask = cv2.dilate(scribble_mask, kernel, iterations=1)
    
    # Ensure scribble stays inside the instance mask
    scribble_mask = scribble_mask & mask
    
    return scribble_mask


def points_to_coco_format(points: List[Tuple[int, int]]) -> List[float]:
    """
    Convert list of (x, y) points to COCO-style flat list [x1, y1, x2, y2, ...].
    
    Args:
        points: List of (x, y) coordinates
    
    Returns:
        Flat list of coordinates
    """
    flat_list = []
    for x, y in points:
        flat_list.extend([float(x), float(y)])
    return flat_list


def polylines_to_coco_format(polylines: List[List[Tuple[int, int]]]) -> List[List[float]]:
    """
    Convert polylines to COCO-style format.
    Each polyline becomes a separate list of coordinates.
    
    Args:
        polylines: List of polylines
    
    Returns:
        List of coordinate lists
    """
    result = []
    for polyline in polylines:
        if polyline:
            coords = points_to_coco_format(polyline)
            if coords:
                result.append(coords)
    return result


def ensure_inside_mask(
    points: List[Tuple[int, int]], 
    mask: np.ndarray
) -> List[Tuple[int, int]]:
    """
    Filter points to ensure they are strictly inside the instance mask.
    
    Args:
        points: List of (x, y) coordinates
        mask: Binary instance mask
    
    Returns:
        Filtered list of points inside the mask
    """
    h, w = mask.shape
    inside_points = []
    
    for x, y in points:
        if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
            inside_points.append((x, y))
    
    return inside_points


def erode_mask_for_interior(mask: np.ndarray, erosion_size: int = 3) -> np.ndarray:
    """
    Erode the mask to get interior region for scribble placement.
    This ensures scribbles don't touch the boundary.
    
    Args:
        mask: Binary instance mask
        erosion_size: Size of erosion kernel
    
    Returns:
        Eroded mask
    """
    if mask.sum() == 0:
        return mask
    
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (erosion_size, erosion_size)
    )
    eroded = cv2.erode(mask, kernel, iterations=1)
    
    # If erosion removes everything, use original
    if eroded.sum() == 0:
        return mask
    
    return eroded


def generate_scribble_for_annotation(
    segmentation: List[List[float]],
    height: int,
    width: int,
    target_coverage: float = 0.03,
    dilation_radius: int = 2,
    min_points: int = 5,
    max_points: int = 80,
    use_medial_axis: bool = False
) -> List[List[float]]:
    """
    Generate scribble annotation for a single instance.
    
    Args:
        segmentation: COCO-style segmentation polygon
        height: Image height
        width: Image width
        target_coverage: Target coverage (0.01-0.05)
        dilation_radius: Dilation for simulating human stroke
        min_points: Minimum points in scribble
        max_points: Maximum points in scribble
        use_medial_axis: Use medial axis instead of skeletonization
    
    Returns:
        Scribble in COCO-style format (list of polyline coordinate lists)
    """
    # Step 1: Convert segmentation to mask
    mask = polygon_to_mask(segmentation, height, width)
    
    if mask.sum() == 0:
        return []
    
    instance_area = int(mask.sum())
    
    # Step 2: Erode mask to get interior (avoid boundary)
    interior_mask = erode_mask_for_interior(mask, erosion_size=5)
    
    # Step 3: Extract skeleton from interior
    if use_medial_axis:
        skeleton = extract_medial_axis_skeleton(interior_mask)
    else:
        skeleton = extract_skeleton(interior_mask)
    
    # If skeleton is empty, try with original mask
    if skeleton.sum() == 0:
        if use_medial_axis:
            skeleton = extract_medial_axis_skeleton(mask)
        else:
            skeleton = extract_skeleton(mask)
    
    if skeleton.sum() == 0:
        # Fallback: use centroid area
        return generate_fallback_scribble(mask, target_coverage, min_points)
    
    # Step 4: Get skeleton points
    skeleton_points = get_skeleton_points(skeleton)
    
    if not skeleton_points:
        return generate_fallback_scribble(mask, target_coverage, min_points)
    
    # Step 5: Subsample to achieve sparse coverage
    subsampled = subsample_skeleton_points(
        skeleton_points,
        target_coverage=target_coverage,
        instance_area=instance_area,
        min_points=min_points,
        max_points=max_points
    )
    
    # Step 6: Ensure all points are inside the mask
    inside_points = ensure_inside_mask(subsampled, mask)
    
    if not inside_points:
        return generate_fallback_scribble(mask, target_coverage, min_points)
    
    # Step 7: Create connected polylines
    polylines = create_connected_polylines(inside_points, max_gap=15)
    
    if not polylines:
        # If no connected polylines, return as single list
        return [points_to_coco_format(inside_points)]
    
    # Step 8: Convert to COCO format
    scribble = polylines_to_coco_format(polylines)
    
    # Verify coverage
    if scribble:
        total_points = sum(len(s) // 2 for s in scribble)
        coverage = total_points / instance_area if instance_area > 0 else 0
        
        # If coverage is too low, add more points
        if coverage < 0.005 and len(inside_points) > total_points:
            return [points_to_coco_format(inside_points)]
    
    return scribble if scribble else []


def generate_fallback_scribble(
    mask: np.ndarray,
    target_coverage: float = 0.03,
    min_points: int = 5
) -> List[List[float]]:
    """
    Fallback scribble generation when skeleton fails.
    Uses random interior points near the centroid.
    
    Args:
        mask: Binary instance mask
        target_coverage: Target coverage
        min_points: Minimum number of points
    
    Returns:
        Scribble in COCO format
    """
    if mask.sum() == 0:
        return []
    
    # Find all interior points
    interior_points = np.where(mask > 0)
    if len(interior_points[0]) == 0:
        return []
    
    # Get centroid
    cy = int(np.mean(interior_points[0]))
    cx = int(np.mean(interior_points[1]))
    
    # Get points near centroid
    all_points = list(zip(interior_points[1].tolist(), interior_points[0].tolist()))
    
    # Sort by distance to centroid
    all_points.sort(key=lambda p: (p[0] - cx)**2 + (p[1] - cy)**2)
    
    # Take points near center
    num_points = max(min_points, int(len(all_points) * target_coverage))
    num_points = min(num_points, len(all_points), 50)
    
    selected = all_points[:num_points]
    
    if selected:
        return [points_to_coco_format(selected)]
    
    return []


def verify_scribble(
    scribble: List[List[float]],
    segmentation: List[List[float]],
    height: int,
    width: int
) -> Dict[str, Any]:
    """
    Verify that generated scribble meets requirements.
    
    Args:
        scribble: Generated scribble
        segmentation: Original segmentation
        height: Image height
        width: Image width
    
    Returns:
        Dictionary with verification results
    """
    mask = polygon_to_mask(segmentation, height, width)
    instance_area = int(mask.sum())
    
    if not scribble or instance_area == 0:
        return {
            'valid': False,
            'num_points': 0,
            'coverage': 0.0,
            'inside_mask': True
        }
    
    # Count total points
    total_points = sum(len(s) // 2 for s in scribble)
    
    # Check if all points are inside mask
    all_inside = True
    for polyline in scribble:
        for i in range(0, len(polyline), 2):
            x, y = int(polyline[i]), int(polyline[i+1])
            if 0 <= x < width and 0 <= y < height:
                if mask[y, x] == 0:
                    all_inside = False
                    break
    
    coverage = total_points / instance_area
    
    return {
        'valid': total_points > 0 and all_inside,
        'num_points': total_points,
        'coverage': coverage,
        'inside_mask': all_inside
    }


def process_json_file(
    input_path: str,
    output_path: str,
    target_coverage: float = 0.03,
    dilation_radius: int = 2,
    use_medial_axis: bool = False,
    verify: bool = True
) -> Dict[str, Any]:
    """
    Process a RefHuman JSON file and add scribble annotations.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        target_coverage: Target scribble coverage (0.01-0.05)
        dilation_radius: Dilation radius for scribble
        use_medial_axis: Use medial axis instead of skeletonization
        verify: Whether to verify generated scribbles
    
    Returns:
        Statistics dictionary
    """
    print(f"\n{'='*60}")
    print(f"Processing: {input_path}")
    print(f"{'='*60}")
    
    # Load JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Build image lookup for dimensions
    image_lookup = {}
    for img in data['images']:
        image_lookup[img['id']] = {
            'height': img['height'],
            'width': img['width']
        }
    
    # Statistics
    stats = {
        'total_annotations': len(data['annotations']),
        'successful': 0,
        'failed': 0,
        'empty_segmentation': 0,
        'avg_coverage': 0.0,
        'avg_points': 0.0
    }
    
    coverages = []
    point_counts = []
    
    # Process each annotation
    print(f"\nGenerating scribbles for {len(data['annotations'])} annotations...")
    
    for ann in tqdm(data['annotations'], desc="Generating scribbles"):
        img_info = image_lookup.get(ann['image_id'], {})
        height = img_info.get('height', 480)
        width = img_info.get('width', 640)
        
        segmentation = ann.get('segmentation', [])
        
        if not segmentation or (isinstance(segmentation, list) and len(segmentation) == 0):
            stats['empty_segmentation'] += 1
            ann['scribble'] = []
            continue
        
        # Handle RLE segmentation (skip for now, focus on polygon)
        if isinstance(segmentation, dict):
            stats['empty_segmentation'] += 1
            ann['scribble'] = []
            continue
        
        # Generate scribble
        try:
            scribble = generate_scribble_for_annotation(
                segmentation=segmentation,
                height=height,
                width=width,
                target_coverage=target_coverage,
                dilation_radius=dilation_radius,
                use_medial_axis=use_medial_axis
            )
            
            if scribble:
                ann['scribble'] = scribble
                stats['successful'] += 1
                
                if verify:
                    verification = verify_scribble(scribble, segmentation, height, width)
                    coverages.append(verification['coverage'])
                    point_counts.append(verification['num_points'])
            else:
                ann['scribble'] = []
                stats['failed'] += 1
                
        except Exception as e:
            ann['scribble'] = []
            stats['failed'] += 1
    
    # Calculate averages
    if coverages:
        stats['avg_coverage'] = np.mean(coverages)
        stats['avg_points'] = np.mean(point_counts)
    
    # Save output
    print(f"\nSaving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    
    # Print statistics
    print(f"\n{'='*60}")
    print("STATISTICS")
    print(f"{'='*60}")
    print(f"Total annotations:     {stats['total_annotations']}")
    print(f"Successful:            {stats['successful']} ({100*stats['successful']/stats['total_annotations']:.1f}%)")
    print(f"Failed:                {stats['failed']} ({100*stats['failed']/stats['total_annotations']:.1f}%)")
    print(f"Empty segmentation:    {stats['empty_segmentation']}")
    print(f"Average coverage:      {stats['avg_coverage']*100:.2f}%")
    print(f"Average points:        {stats['avg_points']:.1f}")
    print(f"{'='*60}\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Generate scribble annotations from segmentation polygons'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input JSON file path'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output JSON file path'
    )
    parser.add_argument(
        '--coverage', '-c',
        type=float,
        default=0.03,
        help='Target coverage (0.01-0.05, default: 0.03)'
    )
    parser.add_argument(
        '--dilation', '-d',
        type=int,
        default=2,
        help='Dilation radius (1-3, default: 2)'
    )
    parser.add_argument(
        '--medial-axis',
        action='store_true',
        help='Use medial axis transform instead of skeletonization'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip verification of generated scribbles'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    if not 0.01 <= args.coverage <= 0.1:
        print(f"Warning: Coverage {args.coverage} outside recommended range (0.01-0.05)")
    
    # Process
    process_json_file(
        input_path=args.input,
        output_path=args.output,
        target_coverage=args.coverage,
        dilation_radius=args.dilation,
        use_medial_axis=args.medial_axis,
        verify=not args.no_verify
    )


if __name__ == '__main__':
    main()
