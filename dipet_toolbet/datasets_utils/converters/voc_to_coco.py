import xml.etree.ElementTree as ET

from tqdm import tqdm
from pathlib import Path


from dipet_toolbet.datasets_utils.bbox_utils import convert_bbox
from dipet_toolbet.datasets_utils.converters.coco_base import CocoBase


class VocToCoco(CocoBase):
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

    def parse_dir(self, directory, silent=False):
        directory = Path(directory)

        if silent:
            data = directory.glob('*.xml')
        else:
            data = tqdm(list(directory.glob('*.xml')))
        for xml_path in data:
            self.parse_file(xml_path)


def parse_voc_to_coco(path, silent=False):
    path = Path(path)
    dataset = VocToCoco()

    if path.is_dir():
        dataset.parse_dir(path, silent)
    else:
        dataset.parse_file(path)

    return dataset.dataset


def parse_pascal_to_coco(path, silent=False):
    return parse_voc_to_coco(path, silent)


__all__ = [
    'parse_voc_to_coco',
    'parse_pascal_to_coco',
    'VocToCoco',
]
