FROM redis:latest

WORKDIR /data
COPY seed.txt .
COPY seed_redis.sh .
RUN chmod +x seed_redis.sh

CMD ["./seed_redis.sh"]