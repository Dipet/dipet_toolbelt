SUPPORTED_FORMATS = [
    'coco',
    'pascal',
    'voc',
    'xyxy',
    'xyhw',
]


def convert_bbox(bbox, src_format, dst_format='xyxy'):
    if src_format == dst_format:
        return bbox

    if src_format in ['coco', 'xyhw']:
        (x1, y1, h, w), tail = bbox[:4], bbox[4:]
        x2 = x1 + w
        y2 = y1 + h
    elif src_format in ['pascal', 'voc', 'xyxy']:
        (x1, y1, x2, y2), tail = bbox[:4], bbox[4:]
    else:
        raise ValueError(f'Unsupported src format: {src_format}')

    if dst_format in ['coco', 'xyhw']:
        return [x1, y1, x2 - x1, y2 - y1] + list(tail)
    elif dst_format in ['pascla', 'voc', 'xyxy']:
        return [x1, y1, x2, y2] + list(tail)

    raise ValueError(f'Unsupported dst format: {dst_format}')
