#!/bin/bash

# The purpose of this custom AppRun script is
# to allow symlinking the AppImage and invoking
# the corresponding binary depending on which
# symlink was used to invoke the AppImage

HERE="$(dirname "$(readlink -f "${0}")")"

if [ ! -z $APPIMAGE ] ; then
  BINARY_NAME=$(basename "$ARGV0")
  if [ -e "$HERE/$BINARY_NAME" ] ; then
    exec "$HERE/$BINARY_NAME" "$@"
  else
    exec "$HERE/VC" "$@"
  fi
else
  exec "$HERE/VC" "$@"
fi
