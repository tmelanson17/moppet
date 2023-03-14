# moppet
A robot that uses ceiling tiles to navigate a room. 

Youtube Video demonstrating the code: https://youtu.be/JS_mKLHqY1s

## Description
- The `pygame_tutorial.py` file creates a sim environment which visualizes the vehicle in the environment, the particle filter, as well as basic statistics on "sensor" measurements and expected location.
- `process_image.py` converts the image into a tile map, and does the SURF output comparison used to determine likelihood in the particle filter

## Requirements
- Python 3
- anaconda

## Install & run (terminal)
```
conda env create -f environment.yml
python pygame_tutorial.py
```

## TODO (Near term)

### Expand implementation
- Adjust the sim and filter to account for camera orientation changes
- Have the ability to use the filter with a Pi robot.
- Create a controller for the robot so it can drive itself
### Expand testing and visualization
- Create a unit test to demonstrate the failure rate of the particle filter
- Create a marker for where the particle filter predicted location is and variance
### Performance enhancements
- Limit the number of SURF features detected, as it slows down with highly detailed tiles.
### Reorg the code for better understanding.
- Rename `pygame_tutorial.py`
- Remove the `Corridor` class, it only serves to define the active spaces in the sim

