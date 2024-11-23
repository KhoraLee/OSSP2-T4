FROM python:3.11-slim

WORKDIR /usr/RLTrader

COPY requirements.txt .
COPY src src
COPY data data

RUN pip install --no-cache-dir -r requirements.txt

# 로케일 설정
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y locales
RUN sed -i -e 's/# ko_KR.UTF-8 UTF-8/ko_KR.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=ko_KR.UTF-8

ENV LANG=ko_KR.UTF-8
RUN locale-gen ko_KR.UTF-8
ENV LANG=ko_KR.UTF-8
ENV LANGUAGE=ko_KR.UTF-8
ENV LC_ALL=ko_KR.UTF-8

ENV DOCKER=1

#CMD ["/bin/bash"]
ENTRYPOINT [ "python", "src/main.py"]