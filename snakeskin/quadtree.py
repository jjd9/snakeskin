import numpy as np 
from PIL import Image, ImageDraw

MAX_DEPTH = 8
DETAIL_THRESHOLD = 13
SIZE_MULT = 1

def average_colour(image):
    # convert image to np array
    image_arr = np.asarray(image)
    return np.mean(image_arr.reshape(-1,3).astype(float), axis=0).astype(np.uint8)

def weighted_average(hist):
    total = np.sum(hist)
    error = value = 0

    if total > 0:
        indexes = np.arange(0, len(hist))
        value = np.sum(indexes * hist) / total
        error = np.sum(hist * (value - indexes)**2 ) / total
        error = np.sqrt(error)

    return error

def get_detail(hist):
    red_detail = weighted_average(hist[:256])
    green_detail = weighted_average(hist[256:512])
    blue_detail = weighted_average(hist[512:768])

    detail_intensity = red_detail * 0.2989 + green_detail * 0.5870 + blue_detail * 0.1140

    return detail_intensity

class Quadrant():
    def __init__(self, image, bbox, depth):
        self.bbox = bbox
        self.depth = depth
        self.children = None
        self.leaf = False

        # crop image to quadrant size
        image = image.crop(bbox)
        hist = image.histogram()

        self.detail = get_detail(hist)
        self.colour = average_colour(image)

    def split_quadrant(self, image):
        left, top, width, height = self.bbox

        # get the middle coords of bbox
        middle_x = left + (width - left) / 2
        middle_y = top + (height - top) / 2

        # split root quadrant into 4 new quadrants
        upper_left = Quadrant(image, (left, top, middle_x, middle_y), self.depth+1)
        upper_right = Quadrant(image, (middle_x, top, width, middle_y), self.depth+1)
        bottom_left = Quadrant(image, (left, middle_y, middle_x, height), self.depth+1)
        bottom_right = Quadrant(image, (middle_x, middle_y, width, height), self.depth+1)

        # add new quadrants to root children
        self.children = [upper_left, upper_right, bottom_left, bottom_right]

class QuadTree():
    def __init__(self, _image):
        if not isinstance(_image, Image.Image):
            image = Image.fromarray(_image[:,:,::-1])
        else:
            image = _image

        self.width, self.height = image.size 

        # keep track of max depth achieved by recursion
        self.max_depth = 0

        # start compression
        self.start(image)
    
    def start(self, image):
        # create initial root
        self.root = Quadrant(image, image.getbbox(), 0)
        
        # build quadtree
        self.build_iterative(self.root, image)
        # self.build(self.root, image)

    def build(self, root, image):
        if root.depth >= MAX_DEPTH or root.detail <= DETAIL_THRESHOLD:
            if root.depth > self.max_depth:
                self.max_depth = root.depth

            # assign quadrant to leaf and stop recursing
            root.leaf = True
            return 
        
        # split quadrant if there is too much detail
        root.split_quadrant(image)

        for children in root.children:
            self.build(children, image)

    def build_iterative(self, root, image):
        stack = []
        stack.append(root)
        count = 1
        
        while count:
            node = stack.pop()
            count-=1
            if node.depth >= MAX_DEPTH or node.detail <= DETAIL_THRESHOLD:
                if node.depth > self.max_depth:
                    self.max_depth = node.depth

                # assign quadrant to leaf and stop recursing
                node.leaf = True
                continue
            
            # split quadrant if there is too much detail
            node.split_quadrant(image)

            for child in node.children:
                stack.append(child)
                count+=1

    def create_image(self, custom_depth, show_lines=False):
        # create blank image canvas
        image = Image.new('RGB', (self.width, self.height))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, self.width, self.height), (0, 0, 0))

        leaf_quadrants = self.get_leaf_quadrants(custom_depth)

        # draw rectangle size of quadrant for each leaf quadrant
        for quadrant in leaf_quadrants:
            if show_lines:
                draw.rectangle(quadrant.bbox, quadrant.colour, outline=(0, 0, 0))
            else:
                draw.rectangle(quadrant.bbox, quadrant.colour)

        return image

    def get_leaf_quadrants(self, depth):
        if depth > self.max_depth:
            raise ValueError('A depth larger than the trees depth was given')

        quandrants = []

        # search recursively down the quadtree
        self.recursive_search(self, self.root, depth, quandrants.append)

        return quandrants

    def recursive_search(self, tree, quadrant, max_depth, append_leaf):
        # append if quadrant is a leaf
        if quadrant.leaf == True or quadrant.depth == max_depth:
            append_leaf(quadrant)

        # otherwise keep recursing
        elif quadrant.children != None:
            for child in quadrant.children:
                self.recursive_search(tree, child, max_depth, append_leaf)

    def create_gif(self, file_name, duration=1000, loop=0, show_lines=False):
        gif = []
        end_product_image = self.create_image(self.max_depth, show_lines=show_lines)

        for i in range(self.max_depth):
            image = self.create_image(i, show_lines=show_lines)
            gif.append(image)

        # add extra images at end
        for _ in range(4):
            gif.append(end_product_image)

        gif[0].save(
            file_name,
            save_all=True,
            append_images=gif[1:],
            duration=duration, loop=loop)
