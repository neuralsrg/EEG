#!/bin/bash

# creating curl request to get file stored in git-lfs

#USAGE:
# 1. make .sh file executable: $ chmod +x lfs_request
# 2. execute $ ./lfs_request.sh
# 3. specify pointer file 
# 4. specify organisation (github username)
# 5. specify repo (if it contains whitespaces, replace them with '%20')

# request:

# curl -X POST \
# -H "Accept: application/vnd.git-lfs+json" \
# -H "Content-type: application/json" \
# -d '{"operation": "download", "transfer": ["basic"], "objects": [{"oid": "{sha}", "size": {size}}]}' \
# https://github.com/{organisation}/{repository}.git/info/lfs/objects/batch

read -p "enter git-lfs pointer file name: " filename

sha="$(cat $filename | grep oid | cut -d ":" -f 2)"
size="$(cat $filename | grep size | cut -d " " -f 2)"

read -p "enter organization (e.g. neuralsrg): " org
read -p "enter repo: " repo

part1="{\"operation\": \"download\", \"transfer\": [\"basic\"], \"objects\": [{\"oid\": \""
part2="\", \"size\": "
part3="}]}"

echo Creating POST request...

curl -X POST \
-H "Accept: application /vnd.git-lfs+json" \
-H "Content-type: application/json" \
-d "${part1}${sha}${part2}${size}${part3}" \
https://github.com/"${org}"/"${repo}".git/info/lfs/objects/batch
