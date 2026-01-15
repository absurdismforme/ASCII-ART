import cv2 as cv
import numpy as np


def load_assets(file, tile_size):
    img = cv.imread(file, cv.IMREAD_GRAYSCALE)
    assert img is not None, f"File {file} could not be read"

    texture = cv.imread("textures/ASCII.png", cv.IMREAD_GRAYSCALE)
    texture_edge = cv.imread("textures/edgesASCII.png", cv.IMREAD_GRAYSCALE)
    if texture_edge is None:
        texture_edge = texture.copy()

    h, w = img.shape
    crop_h, crop_w = h - (h % tile_size), w - (w % tile_size)
    img = img[:crop_h, :crop_w]

    return img, texture, texture_edge, crop_h, crop_w


def get_luminance_map(img, tile_size, crop_w, crop_h):
    img_downscaled = cv.resize(
        img, (crop_w // tile_size, crop_h // tile_size), interpolation=cv.INTER_AREA
    )
    # Quantize 0-255 into 0-9 levels for the 10 ASCII characters
    img_quantised = np.floor(img_downscaled.astype(np.float32) * 10 / 256).astype(
        np.uint8
    )

    return cv.resize(img_quantised, (crop_w, crop_h), interpolation=cv.INTER_NEAREST)


def get_edge_map(img, tile_size, crop_w, crop_h, min_magnitude):
    # Sobel operators for gradient calculation
    sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    angle_deg = np.degrees(np.arctan2(sobel_y, sobel_x)) % 180

    # Classify edge direction per pixel
    edge_type = np.zeros_like(img, dtype=np.uint8)
    edge_threshold = np.percentile(magnitude, 98)
    mask = magnitude > edge_threshold

    edge_type[mask & ((angle_deg >= 67.5) & (angle_deg < 112.5))] = 1  # Vertical
    edge_type[mask & ((angle_deg >= 0) & (angle_deg < 22.5) | (angle_deg >= 157.5))] = (
        2  # Horizontal
    )
    edge_type[mask & ((angle_deg >= 22.5) & (angle_deg < 67.5))] = 3  # Diagonal 1
    edge_type[mask & ((angle_deg >= 112.5) & (angle_deg < 157.5))] = 4  # Diagonal 2

    # Block analysis: Find the dominant edge type in each tile
    sh, sw = crop_h // tile_size, crop_w // tile_size
    blocks = edge_type.reshape(sh, tile_size, sw, tile_size).transpose(0, 2, 1, 3)
    mag_blocks = magnitude.reshape(sh, tile_size, sw, tile_size).transpose(0, 2, 1, 3)

    avg_mag = np.mean(mag_blocks, axis=(2, 3))
    counts = np.array(
        [np.sum(blocks == t, axis=(2, 3)) for t in range(1, 5)]
    ).transpose(1, 2, 0)

    edge_small = (np.argmax(counts, axis=2) + 1).astype(np.uint8)
    max_counts = np.max(counts, axis=2)

    # Filter out weak edges
    edge_small[(avg_mag < min_magnitude) | (max_counts < 5)] = 0

    # Upscale to original size
    edge_upscaled = cv.resize(
        edge_small, (crop_w, crop_h), interpolation=cv.INTER_NEAREST
    )
    return edge_upscaled


def compose_ascii(lum_map, edge_map, texture, texture_edge, tile_size, h, w):
    y, x = np.indices((h, w))
    ly, lx = y % tile_size, x % tile_size

    # Calculate offsets in the texture atlas
    output_lum = texture[ly, lx + (lum_map[y, x] * tile_size)]
    output_edge = texture_edge[ly, lx + (edge_map[y, x] * tile_size)]

    # Use edge texture where an edge is detected, otherwise use luminance texture
    return np.where(edge_map > 0, output_edge, output_lum)


def shader(file_path, output_path="images/ASCII_output.png"):
    TILE_SIZE = 8
    MIN_MAG = 50

    img, tex, tex_e, h, w = load_assets(file_path, TILE_SIZE)

    lum_map = get_luminance_map(img, TILE_SIZE, w, h)
    edge_map = get_edge_map(img, TILE_SIZE, w, h, MIN_MAG)

    final_img = compose_ascii(lum_map, edge_map, tex, tex_e, TILE_SIZE, h, w)

    cv.imwrite(output_path, final_img)
    return True
