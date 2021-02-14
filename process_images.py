import pygame
import cv2
import numpy as np
from enum import Enum
from sklearn.cluster import KMeans
from scipy.stats import chisquare

EXAMPLE_CEILING = 'ceiling.jpg'

def get_image(display_width, display_height, filename=EXAMPLE_CEILING):
    ceiling = cv2.imread(filename)
    ceiling = cv2.flip(ceiling, 0)
    ceiling = cv2.rotate(ceiling, cv2.ROTATE_90_CLOCKWISE)
    ceiling = cv2.resize(ceiling, (display_height, display_width))
    return ceiling

class Tile():
    def __init__(self, loc, image):
        self.loc = loc
        self.set_img(image)

    def draw_tile(self):
        return self.surface

    def set_img(self, img):
        self.image = img
        self.surface = pygame.surfarray.make_surface(img)
        self.surface = self.surface.convert()


class TileMap():
    '''
      TODO: What if the image isn't rectangular? (i.e. has inlets / walls)
    '''
    def __init__(self, img, tile_width=60, tile_height=60):
        self.tile_width=tile_width
        self.tile_height=tile_height
        self.n_tiles_width = int(np.ceil(img.shape[0] / self.tile_width))
        self.n_tiles_height = int(np.ceil(img.shape[1] / self.tile_height))
        self._tiles = []
        self.img = np.copy(img)
        for i in range(self.n_tiles_width):
            column = []
            for j in range(self.n_tiles_height):
                column.append(Tile((i, j), img[(self.tile_width*i):(self.tile_width*(i+1)), (self.tile_height*j):(self.tile_height*(j+1))]))
            self._tiles.append(column)


    def convert_index_to_location(self, location):
        return (location[0] * self.tile_width, location[1] * self.tile_height) 

    def draw_tiles(self, display):
        for i in range(self.n_tiles_width):
            for j in range(self.n_tiles_height):
                display.blit(self._tiles[i][j].draw_tile(), self.convert_index_to_location((i, j)))

    def get_tile(self, i, j):
        return self._tiles[i][j]

    def set_tile_img(self, i, j, img):
        self._tiles[i][j].set_img(img)

    def make_tile(self, x, y):
        x = int(x)
        y = int(y)
        return  Tile((x,y), 
                    self.img[
                    (x - self.tile_width//2):(x + self.tile_width//2),
                    (y - self.tile_height//2):(y + self.tile_height//2) ]
                )


class DescriptorComparison:
    def __init__(self, hessian_threshold, img, n_clusters=15, distance_threshold=0.1):
        self.detector = cv2.xfeatures2d.SURF_create(hessian_threshold)
        self.n_clusters=n_clusters
        kp, self.descriptor_list = self.detector.detectAndCompute(img, None)
        self.kmeans = KMeans(n_clusters=self.n_clusters).fit(self.descriptor_list)
        self.distance_threshold=distance_threshold

    def image_similarity(self, observed, expected):
        kp1, des1 = self.detector.detectAndCompute(observed, None)
        kp2, des2 = self.detector.detectAndCompute(expected, None)
        epsilon = 1e-5
        # If there are no descriptors in observed image or expected image, then mark as similar 
        if (des2 is None or len(des2) == 0) and (des1 is None or len(des1) == 0):
            return 1.0
        elif des1 is None or len(des1) == 0:
            return epsilon
        elif des2 is None or len(des2) == 0:
            return epsilon
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        # Match descriptors.
        matches = bf.match(des1,des2)
        if len(matches) == 0:
            return epsilon
        average_dist = sum([m.distance for m in matches]) / len(matches)
        return np.exp(-average_dist)+epsilon

    def draw_keypoints(self, img):
        # Find keypoints and descriptors directly 
        kp, des = self.detector.detectAndCompute(img,None)
        img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
        return img2

    def likelihood(self, tilemap, expected, data_observed):
        img_observed = data_observed
        img_expected = tilemap.make_tile(int(expected[0]), int(expected[1])).image
        return self.image_similarity(img_observed, img_expected)
        

if __name__ == '__main__':

    # Add display parameters
    DISPLAY_WIDTH = 780
    DISPLAY_HEIGHT = 600
    gameDisplay = pygame.display.set_mode((DISPLAY_WIDTH,DISPLAY_HEIGHT))
    pygame.display.set_caption('A bit Racey')
    clock = pygame.time.Clock()

    gameDisplay = pygame.display.set_mode((DISPLAY_WIDTH,DISPLAY_HEIGHT))
    pygame.display.set_caption('tilemap')

    ceiling = get_image(DISPLAY_WIDTH, DISPLAY_HEIGHT)
    tilemap = TileMap(ceiling) 

    # Set Hessian threshold to 300
    desc = DescriptorComparison(300, ceiling)
    observed_img = np.copy(tilemap.get_tile(1, 1).image)
    for i in range(tilemap.n_tiles_width):
        for j in range(tilemap.n_tiles_height):
            expected_tile = tilemap.get_tile(i, j)
            # Determine similarity between observed and expected tile
            similarity = desc.image_similarity(observed_img, expected_tile.image)
            print(similarity)
            expected_tile.set_img(expected_tile.image * similarity)


    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
        white = (255,255,255)
        gameDisplay.fill(white)
        tilemap.draw_tiles(gameDisplay)
        pygame.display.update()
        clock.tick(60)


    pygame.quit()
    quit()
