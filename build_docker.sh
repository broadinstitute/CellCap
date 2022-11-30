#!/bin/bash

VERSION=$(cat VERSION)

docker build --file docker/Dockerfile . --tag perturbations:${VERSION}
