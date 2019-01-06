FROM tensorflow/tensorflow:1.12.0-gpu
MAINTAINER Taekmin Kim <tantara.tm@gmail.com>

ENV LC_ALL C
ENV APP_PATH /base

# for docker hub
# COPY . $APP_PATH
# for development
RUN mkdir -p $APP_PATH

WORKDIR $APP_PATH

RUN apt-get update
RUN apt-get install -y screen htop git vim zip

RUN pip install easydict
RUN pip install h5py keras tqdm==4.19.9

# konlpy
RUN apt-get install -y g++ default-jdk python-dev python3-dev
RUN pip install konlpy
#RUN bash -c "curl -sL https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh | bash -"
RUN apt-get purge -y --auto-remove g++ python-dev python3-dev

# locale ko_KR.UTF-8
RUN apt-get install -y locales && locale-gen ko_KR.UTF-8
ENV LC_ALL ko_KR.UTF-8

# ssh
RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:kakao18' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

RUN apt-get clean && rm -rf /var/cache/apt/archives && rm -rf /var/lib/apt/lists

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
