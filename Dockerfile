FROM pytorch/pytorch:2.5.0-cuda12.8.0-cudnn9-runtime

RUN apt-get update && apt-get install -y ffmpeg

RUN pip install --upgrade pip
RUN pip install torch torchaudio torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN pip install pytorch-lightning
RUN pip install datasets
RUN pip install torchaudio

RUN python3 -m pip install -c constraints.txt huggingface_hub==0.23.2
RUN python3 -m pip install -c constraints.txt git+https://github.com/NVIDIA/NeMo.git@r2.0.0rc0#egg=nemo_toolkit[asr]