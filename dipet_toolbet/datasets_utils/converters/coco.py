import xml.etree.ElementTree as ET

from pathlib import Path

from dipet_toolbet.datasets_utils.bbox_utils import convert_bbox


class VOC2COCO:
    def __init__(self):
        self.cat_id = 0
        self.anno_id = 0
        self.image_id = 0

        self.images = []
        self.categories = []
        self.annotations = []

        self.cat_names = {}
        self.file_names = {}

    @property
    def dataset(self):
        return {
            'images': self.images,
            'annotations': self.annotations,
            'categories': self.categories,
        }

    def _next_cat_id(self):
        self.cat_id += 1
        return self.cat_id

    def _next_image_id(self):
        self.image_id += 1
        return self.image_id

    def _next_anno_id(self):
        self.anno_id += 1
        return self.anno_id

    def add_category(self, name):
        if name in self.cat_names:
            return self.cat_names[name]

        cat_id = self._next_cat_id(),

        category_item = {
            'supercategory': 'None',
            'id': cat_id,
            'name': name,
        }

        self.categories.append(category_item)
        self.cat_names[name] = cat_id

        return cat_id

    def add_img_item(self, file_name, h, w):
        if file_name in self.file_names:
            return self.file_names[file_name]

        if file_name is None:
            raise Exception('Could not find filename tag in xml file.')
        if w is None:
            raise Exception('Could not find width tag in xml file.')
        if h is None:
            raise Exception('Could not find height tag in xml file.')

        image_id = self._next_image_id()
        image_item = {
            'id': image_id,
            'file_name': file_name,
            'width': w,
            'height': h,
        }

        self.images.append(image_item)
        self.file_names[file_name] = image_id

        return image_id

    def add_anno_item(self, image_id, category_id, bbox):
        annotation_id = self._next_anno_id()
        x, y, h, w = bbox

        annotation_item = {
            'segmentation': [],
            'area': h * w,
            'iscrowd': 0,
            'ignore': 0,
            'image_id': image_id,
            'bbox': bbox,
            'category_id': category_id,
            'id': annotation_id
        }

        self.annotations.append(annotation_item)
        return annotation_id

    def _parse_xml_filename(self, elem):
        file_name = elem.text
        if file_name in self.file_names:
            raise Exception('filename duplicated')

        return file_name

    @staticmethod
    def _parse_xml_size(elem: ET.Element):
        h = elem.find('height').text
        w = elem.find('width').text
        c = elem.find('depth').text

        return h, w, c

    @staticmethod
    def _parse_xml_bbox(elem: ET.Element):
        x1 = float(elem.find('xmin').text)
        y1 = float(elem.find('ymin').text)
        x2 = float(elem.find('xmax').text)
        y2 = float(elem.find('ymax').text)
        bbox = x1, y1, x2, y2

        return convert_bbox(bbox, 'voc', 'coco')

    def _parse_xml_object(self, elem: ET.Element):
        name = elem.find('name').text

        bbox = self._parse_xml_bbox(elem.find('bndbox'))

        obj = [name, bbox]
        objects = [obj]
        for part in elem.findall('part'):
            objects += self._parse_xml_object(part)

        return objects

    def parse_xml(self, xml: ET.Element):
        filename = self._parse_xml_filename(xml.find('filename'))
        h, w, c = self._parse_xml_size(xml.find('size'))
        objects = []
        for obj in xml.findall('object'):
            objects += self._parse_xml_object(obj)

        image_id = self.add_img_item(filename, h, w)
        for name, bbox in objects:
            cat_id = self.add_category(name)
            self.add_anno_item(image_id, cat_id, bbox)

    def parse_file(self, path):
        tree = ET.parse(str(path))
        root = tree.getroot()

        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        self.parse_xml(root)

    def parse_dir(self, directory):
        directory = Path(directory)

        for xml_path in directory.glob('*.xml'):
            self.parse_file(xml_path)


def parse_voc_to_coco(path):
    path = Path(path)
    dataset = VOC2COCO()

    if path.is_dir():
        dataset.parse_dir(path)
    else:
        dataset.parse_file(path)

    return dataset.dataset


def parse_pascal_to_coco(path):
    return parse_voc_to_coco(path)


__all__ = [
    'parse_voc_to_coco',
    'parse_pascal_to_coco',
    'VOC2COCO',
]
