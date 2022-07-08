#!/bin/bash

read -p "enter github username: " org
read -p "enter repo: " repo
read -p "enter path within the specified repo: " path

curl https://api.github.com/repos/"${org}"/"${repo}"/contents/"${path}"
