SUPPORTED_FORMATS = [
    'coco',
    'pascal',
    'voc',
    'xyxy',
    'xywh',
    'yolo',
    'xywh_normalized'
]


def normalize_bbox(bbox, height, width):
    """Normalize coordinates of a bounding box. Divide x-coordinates by image width and y-coordinates
    by image height.

    Args:
        bbox (list): Denormalized bounding box `(x_min, y_min, x_max, y_max)`.
        height (int): Image height.
        width (int): Image width.

    Returns:
        list: Normalized bounding box `(x_min, y_min, x_max, y_max)`.

    Raises:
        ValueError: If height or width is less or equal zero

    """
    (x_min, y_min, x_max, y_max), tail = bbox[:4], bbox[4:]

    if height <= 0:
        raise ValueError("Argument height must be positive integer")
    if width <= 0:
        raise ValueError("Argument width must be positive integer")

    x_min, x_max = x_min / width, x_max / width
    y_min, y_max = y_min / height, y_max / height

    return [x_min, y_min, x_max, y_max] + list(tail)


def denormalize_bbox(bbox, height, width):
    """Denormalize coordinates of a bounding box. Multiply x-coordinates by image width and y-coordinates
    by image height. This is an inverse operation for `normalize_bbox`.

    Args:
        bbox (list): Normalized bounding box `(x_min, y_min, x_max, y_max)`.
        height (int): Image height.
        width (int): Image width.

    Returns:
        list: Denormalized bounding box `(x_min, y_min, x_max, y_max)`.

    Raises:
        ValueError: If rows or cols is less or equal zero

    """
    (x_min, y_min, x_max, y_max), tail = bbox[:4], bbox[4:]

    if height <= 0:
        raise ValueError("Argument height must be positive integer")
    if width <= 0:
        raise ValueError("Argument width must be positive integer")

    x_min, x_max = x_min * width, x_max * width
    y_min, y_max = y_min * height, y_max * height

    return (x_min, y_min, x_max, y_max) + list(tail)


def convert_bbox(bbox, src_format, dst_format='xyxy', height=None, width=None):
    if src_format == dst_format:
        return bbox

    if src_format in ['coco', 'xywh']:
        (x1, y1, w, h), tail = bbox[:4], bbox[4:]
        x2 = x1 + w
        y2 = y1 + h
    elif src_format in ['pascal', 'voc', 'xyxy']:
        (x1, y1, x2, y2), tail = bbox[:4], bbox[4:]
    elif src_format in ['yolo', 'xywh_normalized']:
        assert height and width, f'height and width must be >= 1 when src_format={src_format}'
        bbox = denormalize_bbox(bbox, height, width)
        (x, y, w, h), tail = bbox[:4], bbox[4:]
        x1 = x - w / 2 + 1
        x2 = x1 + w
        y1 = y - h / 2 + 1
        y2 = y1 + h
    else:
        raise ValueError(f'Unsupported src format: {src_format}')

    if dst_format in ['coco', 'xywh']:
        return [x1, y1, x2 - x1, y2 - y1] + list(tail)
    elif dst_format in ['pascla', 'voc', 'xyxy']:
        return [x1, y1, x2, y2] + list(tail)
    elif dst_format in ['yolo', 'xywh_normalized']:
        assert height and width, f'height and width must be >= 1 when dst_format={src_format}'
        (x1, y1, x2, y2), tail = bbox[:4], bbox[4:]
        x = (x1 + x2) / 2 - 1
        y = (y1 + y2) / 2 - 1
        width = x2 - x1
        height = y2 - y1
        return normalize_bbox([x, y, width, height] + list(tail), height, width)

    raise ValueError(f'Unsupported dst format: {dst_format}')
