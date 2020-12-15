# moppet
A robot that uses ceiling tiles to navigate a room. 

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

# TODO (Near term)
- Rename `pygame_tutorial.py`
- Remove the `Corridor` class, it only serves to define the active spaces in the sim
- Create a marker for where the particle filter predicted location is and variance
- Create a unit test to demonstrate the failure rate of the particle filter
- Adjust the sim and filter to account for camera orientation changes
- Limit the number of SURF features detected, as it slows down with highly detailed tiles.
- Reorg the code for better understanding.
