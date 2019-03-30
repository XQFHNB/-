# usr/env/bin python 3
# -*- coding:utf-8 -*-
import math

print(math.log2(4))

def calcP(a,b):
    return 0-(a*math.log2(a)+b*math.log2(b))


if __name__ == "__main__":
    a=0.4
    b=0.6
    print("result:",calcP(a,b))
    a1=2.0/3
    b1=1.0/3
    print("result:",0.6*calcP(a1,b1))
    a2=0.5
    b2=0.5
    print("result:",0.8*calcP(a2,b2))
