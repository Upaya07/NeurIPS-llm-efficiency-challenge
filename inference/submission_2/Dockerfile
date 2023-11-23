FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

RUN apt-get update  && apt-get install -y git python3-virtualenv wget 

RUN pip install transformers
RUN pip install torch>=2.0.1 huggingface_hub accelerate sentencepiece optimum py7zr scipy appdirs peft

WORKDIR /workspace
# Setup server requriements
COPY ./fast_api_requirements.txt fast_api_requirements.txt
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

ENV HUGGINGFACE_TOKEN="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"
# ENV HUGGINGFACE_REPO="akjindal53244/peft_model_2"
ENV HUGGINGFACE_REPO="upaya07/model_02"

# Copy over single file server
COPY ./main.py main.py
COPY ./api.py api.py
# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
