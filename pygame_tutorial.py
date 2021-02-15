import pygame
import math
import numpy as np
from enum import Enum
from process_images import TileMap, DescriptorComparison, get_image
# TODO: rotate sensor images

pygame.init()


display_width = 800
display_height = 600

black = (0,0,0)
green = (0, 255, 0)
white = (255,255,255)
red = (255, 0, 0)
blue = (0, 0, 255)

# Define environment as list of active tiles
TILE_WIDTH = 60
TILE_HEIGHT = 60

class MoveVector():
    def __init__(self, dX, dTheta):
        self.dX = dX
        self.dTheta = dTheta

    def move(self, particle, noise=None):
        delta_x = self.dX * -np.sin(particle[2])
        delta_y = self.dX * -np.cos(particle[2])
        if noise is not None:
            delta_x = noise.add_noise(delta_x)
            delta_y = noise.add_noise(delta_y)
        delta_x = np.round(delta_x)
        delta_y = np.round(delta_y)
        new_loc_x = particle[0] + delta_x
        new_loc_y = particle[1] + delta_y
        heading = particle[2] + self.dTheta
        return (new_loc_x, new_loc_y, heading)

class CorridorTile():
    def __init__(self, loc, surface_index):
        self.loc = loc
        self.surface_index = surface_index

    def draw_tile(self):
        surface = pygame.Surface((TILE_WIDTH, TILE_HEIGHT))
        surface.fill(colors[self.surface_index])
        return surface

CORRIDOR_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]]

colors = [green, black, white]

# TODO: Update location

'''
   TODO: Deprecate this class
'''
class Corridor:
    def __init__(self):
        # Tile dimensions for mapping to visual plane.
        self.nrows = len(CORRIDOR_MAP)
        self.ncols = len(CORRIDOR_MAP[0])
        self._tiles = []
        for i in range(self.nrows):
            corridor_row = []
            for j in range(self.ncols):
                corridor_row.append(CorridorTile((i, j), CORRIDOR_MAP[i][j]))
            self._tiles.append(corridor_row)
        # Hardcode the car location to be the first tile.
        self.vehicle_tile = (7, 1)
        self.noise = GaussianNoise(0, 5)

    # Get the ceiling tile of a certain reading.
    def read_ceiling(self, index):
        return self._tiles[index[0]][index[1]].draw_tile()

    # Get the distance of the nearest walls.
    def get_walls(self, index):
        WALL = 0
        i, j = index
        # Get north wall
        wall_i = i
        while wall_i >= 0 and self._tiles[wall_i][j].surface_index != WALL:
            wall_i -= 1
        north_dist = (i - wall_i - 0.5) * TILE_HEIGHT
        north_dist = self.noise.add_noise(north_dist)
        # Get south wall
        wall_i = i
        while wall_i < self.nrows and self._tiles[wall_i][j].surface_index != WALL:
            wall_i += 1
        south_dist = (wall_i - i - 0.5) * TILE_HEIGHT 
        south_dist = self.noise.add_noise(south_dist)
        # Get west wall
        wall_j = j
        while wall_j >= 0 and self._tiles[i][wall_j].surface_index != WALL:
            wall_j -= 1
        west_dist = (j - wall_j - 0.5) * TILE_WIDTH 
        west_dist = self.noise.add_noise(west_dist)
        # Get east wall
        wall_j = j
        while wall_j < self.ncols and self._tiles[i][wall_j].surface_index != WALL:
            wall_j += 1
        east_dist = (wall_j - j - 0.5) * TILE_WIDTH 
        east_dist = self.noise.add_noise(east_dist)
        return (north_dist, west_dist, south_dist, east_dist)

    def draw_tiles(self, display):
        for i in range(len(self._tiles)):
            for j in range(len(self._tiles[0])):
                display.blit(self._tiles[i][j].draw_tile(), convert_index_to_location((i, j)))
        car(convert_index_to_location(self.vehicle_tile))

    def get_vehicle_location(self):
        loc = convert_index_to_location(self.vehicle_tile)
        x, y, heading = loc
        x += TILE_WIDTH * 0.5
        y += TILE_HEIGHT * 0.5
        return x, y


    def viable_locations(self, loc):
        WALL=0
        loc_x, loc_y, heading = loc
        index_i = int(round(loc_x / TILE_WIDTH))
        index_j = int(round(loc_y / TILE_HEIGHT))
        in_bounds = index_i < len(self._tiles) and index_j < len(self._tiles[0]) and index_i > 0 and index_j > 0
        return in_bounds and self._tiles[index_i][index_j] != WALL       

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('A bit Racey')


clock = pygame.time.Clock()
crashed = False

class Direction(Enum):
    NORTH=0
    WEST=90
    SOUTH=180
    EAST=270


class Vehicle:
    def __init__(self, initial_loc):
        self.loc = initial_loc
        self.car_direction = Direction.EAST
        car_img = pygame.image.load('racecar.png')
        car_img = car_img.convert()
        self.car_img = pygame.transform.scale(car_img, (TILE_WIDTH, TILE_HEIGHT))

    def advance_vehicle(self, viable_loc_function, move_integration):
        new_loc = move_integration.move(self.loc)

        if viable_loc_function(new_loc):
            self.loc = new_loc
        else:
            # Move heading
            self.loc = (self.loc[0], self.loc[1], new_loc[2])
        return self.loc

    def draw_car(self, display):
        x, y, heading = self.loc
        car_img_rotated = pygame.transform.rotate(self.car_img, int(heading*180/np.pi))
        display.blit(car_img_rotated, (round(x-TILE_WIDTH/2),round(y-TILE_HEIGHT/2)))

def convert_index_to_location(index):
    return (index[1] * TILE_WIDTH, index[0] * TILE_HEIGHT) 

# Adds Gaussian Noise of mean mu and variance sigma to an input value.
class GaussianNoise:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def add_noise(self, values):
        return np.random.normal(values + self.mu, self.sigma**2)

    def probability(self, x):
        return 1/(self.sigma * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((x - self.mu)/self.sigma)**2)

# Models random noise aroud a radar signal
class RadarModel:
    def __init__(self, sigma):
        self.distr = GaussianNoise(0, sigma)

    def probability(self, x, data):
        # Find euclidean distance
        dist = np.sqrt(np.sum(np.square(x - data), axis=1))
        dist = math.sqrt((data[0] - x[0])**2 + (data[1] - x[1])**2)
        return self.distr.probability(dist)


class Combination:
    '''
      @param distros List of probability distributions to combine
    '''
    def __init__(self, *distros):
        self.distros = distros

    def probability(self, x):
        prob = 1
        for d in distros:
            prob *= d.probability(x)
        return prob

# TODO: Record top 10
class ParticleFilter:
    # @param n_particles Number of particles to create for filter
    # @param likelihood_func List of functions that return the likelihood of being in a location given location and input data
    # TODO: Particle range doesn't work with one specification
    def __init__(self, n_particles, particle_range, likelihood_func, viable_particle_func = None):
        self.n_particles = n_particles
        self.likelihood = likelihood_func
        if viable_particle_func is None:
            self.viable_particle_func = lambda x : True
        else:
            self.viable_particle_func = viable_particle_func
        if np.isscalar(particle_range[0]):
            self.max_locations = particle_range[0]
            self.min_locations = (0, 0)
        else:
            self.max_locations = particle_range[1]
            self.min_locations = particle_range[0]
        self.particles = list()
        for i in range(self.n_particles):
            self.particles.append(self.init_particle(self.max_locations, viable_particle_func, self.min_locations))
        self.particles = np.array(self.particles)
        self.top_10 = self.particles[:10]
        self.move_noise = GaussianNoise(0, 2)

    def init_particle(self, max_locations, viable_particle_func, min_locations=(0,0)):
        viable_particle = False
        loc_y = 0
        loc_x = 0
        while not viable_particle:
            loc_x = int(np.floor(np.random.rand() * (max_locations[0] - min_locations[0]) + min_locations[0]))
            loc_y = int(np.floor(np.random.rand() * (max_locations[1] - min_locations[1]) + min_locations[1]))
            # Add heading
            heading = 0.
            viable_particle = viable_particle_func((loc_x, loc_y, heading))
        return (loc_x, loc_y, heading)

    def update(self, data):
        # TODO: extend the likelihood function to be vectorized)
        likelihoods = list()
        for particle in self.particles:
            likelihoods.append(self.likelihood(particle, data))
        total_prob = sum(likelihoods)
        likelihoods = [l / total_prob for l in likelihoods]
        self.top_10 = self.particles[np.argsort(likelihoods)[-10:][::-1]]
        indices = np.random.choice(self.n_particles, size=self.n_particles, p=likelihoods)
        self.particles = self.particles[indices]

    def move_particles(self, move_integration):
        moved_particles = list()
        def move_particle(particle):
            new_loc_x, new_loc_y, heading = move_integration.move(particle, self.move_noise)
            viable_particle = (self.viable_particle_func((new_loc_x, new_loc_y, heading)))
            if viable_particle:
                particle[0] = new_loc_x
                particle[1] = new_loc_y
            particle[2] = heading
            return particle
        
        self.particles = np.apply_along_axis(move_particle, 1, self.particles)

    def get_average_location(self):
        average_x = sum([x for x,y,h in self.particles])/self.n_particles
        average_y = sum([y for x,y,h in self.particles])/self.n_particles
        average_h = np.arctan2(sum([np.sin(h) for x,y,h in self.particles])/self.n_particles, sum([np.cos(h) for x,y,h in self.particles])/self.n_particles)
        return (average_x, average_y, average_h)


def draw_particle_filter(particle_filter, screen):
    def to_pygame_coord(particle):
        return (int(particle[0]), int(particle[1]))
    for particle in particle_filter.particles:
        pygame.draw.circle(screen, red, to_pygame_coord(particle), 2)
    for particle in particle_filter.top_10:
        pygame.draw.circle(screen, blue, to_pygame_coord(particle), 2)

vehicle = Vehicle((TILE_WIDTH, TILE_HEIGHT, Direction.NORTH.value))
corridor = Corridor()
# Ceiling map is 480 wide x 480 high
ceiling = get_image(480, 480)
tilemap = TileMap(ceiling) 
desc = DescriptorComparison(300, ceiling)
particle_filter = ParticleFilter(
                      n_particles=100, 
                      particle_range=[(0, 0), (display_width/2, display_height/2)], 
                      likelihood_func=lambda particle, datum: desc.likelihood(tilemap, particle, datum), 
                      viable_particle_func=corridor.viable_locations
                  )

# create a font object.
# 1st parameter is the font file
# which is present in pygame.
# 2nd parameter is size of the font
font = pygame.font.Font('freesansbold.ttf', 24)
ceiling_tile=tilemap.make_tile(vehicle.loc[0], vehicle.loc[1])
move_dist=0
dX=0
dTheta=0
while not crashed:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True
        '''
           TODO: Move this inside the Vehicle class
        '''
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                dX = 10         
            elif event.key == pygame.K_s:
                dX = 0
            elif event.key == pygame.K_a:
                dTheta = 10
            elif event.key == pygame.K_d:
                dTheta = -10
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                dX = 0         
            elif event.key == pygame.K_s:
                dX = 0
            elif event.key == pygame.K_a:
                dTheta = 0
            elif event.key == pygame.K_d:
                dTheta = 0    

    '''
      Update continuously
    '''
    move_integration = MoveVector(dX, dTheta * np.pi / 180)
    particle_filter.update(ceiling_tile.image)
    vehicle.advance_vehicle(corridor.viable_locations, move_integration)
    particle_filter.move_particles(move_integration)
    '''
    '''
    gameDisplay.fill(white)
    ceiling_tile=tilemap.make_tile(vehicle.loc[0], vehicle.loc[1])

    # create a text suface object,
    # on which text is drawn on it.
    ceiling_text = font.render('Ceiling tile', True, green, white)
    topn_text = font.render('Top 10', True, green, white)
    gameDisplay.blit(ceiling_text, (480, 80))
    gameDisplay.blit(ceiling_tile.draw_tile(), (540, 120))
    gameDisplay.blit(topn_text, (480, 180))
    x_top = 480
    y_top = 220
    for particle in particle_filter.top_10:
        if x_top+TILE_WIDTH >= display_width:
            x_top = 480
            y_top += TILE_HEIGHT
        gameDisplay.blit(tilemap.make_tile(int(particle[0]), int(particle[1])).draw_tile(), (x_top, y_top))
        x_top += TILE_WIDTH

    walls_definitions = font.render('Wall distances (N, W, S, E):', True, green, white)
    location_definitions = font.render('Vehicle Loc (est., act.):', True, green, white)
    walls = corridor.get_walls(corridor.vehicle_tile)
    rounded_walls = (
            round(walls[0], 2),
            round(walls[1], 2),
            round(walls[2], 2),
            round(walls[3], 2))
    walls_text = font.render(str(rounded_walls), True, green, white)
    est_location_text = font.render(str(particle_filter.get_average_location()), True, green, white)
    true_location_text = font.render(str(vehicle.loc), True, green, white)
    base = 360
    gameDisplay.blit(walls_definitions, (480, base))
    gameDisplay.blit(walls_text, (475, base+40))
    gameDisplay.blit(location_definitions, (480, base+80))
    gameDisplay.blit(est_location_text, (480, base+120))
    gameDisplay.blit(true_location_text, (480, base+160))

    #corridor.draw_tiles(gameDisplay)
    tilemap.draw_tiles(gameDisplay)
    vehicle.draw_car(gameDisplay)
    draw_particle_filter(particle_filter, gameDisplay)


    pygame.display.update()
    clock.tick(30)

pygame.quit()
quit()
