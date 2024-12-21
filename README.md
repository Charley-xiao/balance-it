# balance-it

 Balance It! A Python game for CFD

## Run

### Using CPU

```
python main.py
```

### Using GPU

You need to have a CUDA-compatible GPU and install `cupy` to run the GPU version.

```
python main_gpu.py
```

## Effect

Adjust the Renolds number and see the effect.

Roughly speaking, the higher the Renolds number, the more turbulent the flow is.

![image](shot.png)

The above takes about 60 seconds to emerge, which appears to be Von Karman vortices.

## Acknowledgements

This is the final project of the track "Computational Fluid Dynamics" in 0-1 Winter Camp 2024, Shenzhen.

## Others

The code is written and tested in both Python 3.8 and 3.12, and the following packages are required:

- numpy
- pygame
- cupy (optional, for GPU acceleration)

**DO NOT** let the initial velocity (constants.WIND_VELOCITY_X) be too large, or the simulation will not be able to converge.

If you have any questions or improvements regarding the code, please feel free to submit an issue or pull request.