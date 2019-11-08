#!/usr/bin/env sh
cd api
mvn clean package
java -cp target/api-1.0-SNAPSHOT.jar com.lamsade.App

cd ../
./renderer.R

xdg-open experiment.html &
