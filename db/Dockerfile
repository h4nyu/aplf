FROM postgres:10.5

MAINTAINER Xinyaun Yao <yao.xinyuan@canon.co.jp>

ENV POSTGRES_USER aplf
ENV POSTGRES_PASSWORD aplf
ENV POSTGRES_DB aplf
ENV PGDATA /var/lib/postgresql/data/db-files
COPY ./postgresql.conf /etc/postgresql.conf
RUN chmod o+r /etc/postgresql.conf
COPY ./ /srv
WORKDIR /srv
CMD ["postgres", "-c", "config_file=/etc/postgresql.conf"]
