# Grounded Segment Anything Model


## Installation instructions

Follow the official instructions [here](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main#install-with-docker), but first one must install `nvidia-container-toolkit`:
    
```bash
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && \
    sudo apt-get update
    $ sudo apt-get install -y nvidia-container-toolkit
    (Optional) $ sudo systemctl restart docker
```

Then, from inside the `Grounded-Segment-Anything` repo folder:

```bash
    make build-image # Build the docker container; Only needs to be done once
    make run # Run the docker container
```

The scripts inside the container work fine, but not libraries are installed like `litellm`, `langchain`, etc. Some scripts will need more work. 

Easiest way is to use **vs-code**: `attach to container` and then run the scripts from there.
