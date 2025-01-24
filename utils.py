import math


def computeAngle(a, b, c):
    # TODO(shunxian): how to compute the angle?
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return angle
