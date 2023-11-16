
# TRITON inference server
FROM nvcr.io/nvidia/tritonserver:23.10-py3 AS inference
WORKDIR /models
RUN --mount=type=bind,target=/models,src=/models/

FROM nvcr.io/nvidia/pytorch:23.10-py3 AS training
WORKDIR /src
COPY . /src
RUN pip install --no-cache-dir -r requirements.txt
CMD [ "/bin/bash" ]
