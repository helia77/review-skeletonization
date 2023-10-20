#!/bin/bash

for I in {101..164}
do
	convert raw8/*$I.bmp mask8/*$I.bmp +append concat8/img$I.bmp
done
