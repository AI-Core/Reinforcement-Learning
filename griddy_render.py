from gym.envs.classic_control import rendering
import math

def make_oval(width=10, height=10, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2*math.pi*i / res
        points.append((math.cos(ang)*width/2, math.sin(ang)*height/2))
    if filled:
        return rendering.FilledPolygon(points)
    else:
        return rendering.PolyLine(points, True)
