# Python binding for ScalFMM (experimental)


## Build thz python module

To build the python module:
``` bash
cd ScalFMM
guix time-machine -C python_interface/guix-tools/channels.scm -- shell --pure -m python_interface/guix-tools/manifest.scm -- bash --norc
cmake -B build -DBLA_VENDOR=OpenBLAS -Dscalfmm_BUILD_PYFMM=ON
cmake --build build --target pyfmm
```

Do not forget to position the `PYTHONPATH` environment variable:
``` bash
export PYTHONPATH=build/python_interface/
```

## Tutorial



Finally, you can run the tutorial (this is a translation of `examples/tutorial.cpp` which is a simple minimalist 2D toy example):
  - Run the sequential version (example with 50000)
``` bash
python3 python_interface/tutorial.py --group-size 1 --order 5 --tree-height 3 --random  --number-particles 50000 --center 0 0 --width 2 -vv
```

``` bash
python3 python_interface/tutorial.py --group-size 1 --order 5 --tree-height 3 --input_file ../data/units/test_2d_ref.fma  -vv
```
  - Run the OpenMP version (example with 50000)
``` bash
export OMP_MAX_TASK_PRIORITY=11
python3 python_interface/tutorial.py --group-size 1 --order 5 --tree-height 3 --number-particles 50000 --random --center 0 0 --width 2 -vv --open-mp
```

## Examples

### Read/write particles
