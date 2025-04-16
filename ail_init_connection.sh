#!/bin/bash

set -e
set -u

# Check the existence of used environment variables
TEMP="$HOME $USER $HOST $PASS"

# Initiate the master connection
mkdir -p $HOME/.ssh/ctl
sshpass -p $PASS ssh -Nf -o "StrictHostKeyChecking=accept-new" -o ControlMaster=yes -o ControlPath="$HOME/.ssh/ctl/%L-%r@%h:%p" -o ControlPersist=5m $USER@$HOST