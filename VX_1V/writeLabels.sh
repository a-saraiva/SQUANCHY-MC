#!/bin/bash

for {0..5}
do
  for {0..50..10}
  do
    printf "%d %d\n" $x $y 
  done
done