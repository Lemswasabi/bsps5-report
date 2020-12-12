#!/bin/zsh

for folders in letzdatasetraw/*
do
  for file in $folders/* 
  do
    if [[ ${file: -5} == ".webm" ]]
    then
      ffmpeg -i $file letzdatasetwav/${folders##*/}/${${file##*/}%.*}_$(date +%Y%m%d_%H%M%S).wav
    fi
  done
done
