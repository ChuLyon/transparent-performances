#!/usr/bin/env sh
cd api
mvn clean package

if [ $? -ne 0 ]; then
  exit
fi

java -cp target/api-1.0-SNAPSHOT.jar com.lamsade.App

if [ $? -ne 0 ]; then
  exit
fi

cd ../
./renderer.R

if [ $? -ne 0 ]; then
  exit
fi

xdg-open experiment.html &
