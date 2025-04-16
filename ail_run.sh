#!/bin/bash

set -e
set -u

# Set working directory to the directory of the script
cd "$(dirname "$0")"

source .env
source .env.local

# Check the existence of used environment variables
TEMP="$HOME $WS_PATH $USER $HOST $REMOTE_PROJECT_PATH"

# Parse arguments
cd $WS_PATH
args=$(python3 ./ail_run_args.py $@)
exit_code=$?
if [[ $exit_code -eq 0 && $args == ail_opt_successful_arg_parse=1* ]]; then
  source <(echo "$args")
else
  echo "$args"
  exit $exit_code
fi
cd -

source ail_init_connection.sh

OPT_GROUP=${GROUP:+'--groupmap=*:'"$GROUP"}

OPT_RSYNCIGNORE_LOCAL=""
if [ -f .rsyncignore.local ]; then
  OPT_RSYNCIGNORE_LOCAL="--exclude-from=.rsyncignore.local"
fi

if [ $ail_opt_no_sync = 1 ]; then
  echo "Skipping sync from local to remote"
else
  # Make remote target directory
  ssh -o "ControlPath=$HOME/.ssh/ctl/%L-%r@%h:%p" $USER@$HOST mkdir -p $REMOTE_PROJECT_PATH/$WS_PATH

  # Sync from local to remote
  rsync -e "ssh -o 'ControlPath=$HOME/.ssh/ctl/%L-%r@%h:%p'" -urltvzP --perms --chmod=770 $OPT_GROUP $OPT_RSYNCIGNORE_LOCAL --exclude-from=.rsyncignore $WS_PATH/ $USER@$HOST:$REMOTE_PROJECT_PATH/$WS_PATH
fi

# Run command on remote from $REMOTE_PATH/$WS_PATH
ssh -o "ControlPath=$HOME/.ssh/ctl/%L-%r@%h:%p" $USER@$HOST "cd $REMOTE_PROJECT_PATH/$WS_PATH && python3 -u ail_fe_main.py $@"

# Sync from remote to local
if [ $ail_opt_no_sync = 1 ]; then
  echo "Skipping sync from remote to local"
else
  rsync -e "ssh -o 'ControlPath=$HOME/.ssh/ctl/%L-%r@%h:%p'" -urltvzP $OPT_RSYNCIGNORE_LOCAL --exclude-from=.rsyncignore $USER@$HOST:$REMOTE_PROJECT_PATH/$WS_PATH/ $WS_PATH
fi
