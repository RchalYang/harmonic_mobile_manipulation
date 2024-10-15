 ps aux | grep Cloud | cut -c 10-16 | xargs kill
 ps aux | grep Train | cut -c 10-16 | xargs kill -9
 ps aux | grep Valid | cut -c 10-16 | xargs kill -9
 ps aux | grep Vec | cut -c 10-16 | xargs kill -9
