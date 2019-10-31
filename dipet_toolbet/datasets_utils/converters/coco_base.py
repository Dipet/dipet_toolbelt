class CocoBase:
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

        cat_id = self._next_cat_id()

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
