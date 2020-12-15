# moppet
A robot that uses ceiling tiles to navigate a room. 
- The `pygame_tutorial.py` file creates a 

# Requirements
- Python 3
- anaconda

# Install & run (terminal)
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
