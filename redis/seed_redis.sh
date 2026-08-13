#!/bin/sh

redis-server --daemonize yes
sleep 2

cat seed.txt | redis-cli --pipe

redis-cli save

redis-cli shutdown

rm seed.txt

exec redis-server