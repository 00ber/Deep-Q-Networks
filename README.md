# ml-reinforcement-learning

Python version: 3.7.3


Troubleshooting


- RuntimeError: Polyfit sanity test emitted a warning, most likely due to using a buggy Accelerate backend. If you compiled yourself, more information is available at https://numpy.org/doc/stable/user/building.html#accelerated-blas-lapack-libraries Otherwise report this to the vendor that provided NumPy.
RankWarning: Polyfit may be poorly conditioned

```
$ pip uninstall numpy
$ export OPENBLAS=$(brew --prefix openblas)
$ pip install --no-cache-dir  numpy
```


During grpcio installation 👇
distutils.errors.CompileError: command 'clang' failed with exit status 1
```
CFLAGS="-I/Library/Developer/CommandLineTools/usr/include/c++/v1 -I/opt/homebrew/opt/openssl/include" LDFLAGS="-L/opt/homebrew/opt/openssl/lib" pip3 install grpcio
```


ModuleNotFoundError: No module named 'gym.envs.classic_control.rendering'


#Setup

```
conda install pytorch torchvision -c pytorch
pip install gym-retro
conda install numpy
pip install "gym[atari]==0.21.0"
pip install importlib-metadata==4.13.0
```
