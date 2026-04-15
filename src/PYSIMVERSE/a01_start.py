import time

from pysimverse import Drone

if __name__ == '__main__':
    drone = Drone()
    drone.connect()
    drone.take_off()

    drone.move_up(20)
    time.sleep(3)

    drone.land()
