# gnode


conda create -n gnode py=3.8

Activate cuda-11.7:

export PATH=/usr/local/cuda-11.7/bin:$PATH

export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH

Make the Change Persistent If you want the change to persist across sessions, add the environment variables to your shell configuration file:

For bash, edit ~/.bashrc:
echo 'export PATH=/usr/local/cuda-11.7/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
Reload the configuration:
source ~/.bashrc
