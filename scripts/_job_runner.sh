#!/usr/bin/env bash
set -u

EXIT_FILE=$1
shift

on_signal() {
  printf '%s\n' 143 > "$EXIT_FILE"
  exit 143
}
trap on_signal TERM INT

set +e
"$@"
code=$?
printf '%s\n' "$code" > "$EXIT_FILE"
exit "$code"
