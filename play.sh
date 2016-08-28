#!/bin/sh
rm -f tmp.s16
c2dec 3200 $1 tmp.s16
play --norm=3 tmp.s16
