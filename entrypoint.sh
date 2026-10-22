#!/bin/sh
# One image, two servers. No arguments, or arguments that start with "-": Nemotron Deciffusion through the slim serving
# image's /app/start.sh (unchanged; extra arguments go to start.sh as before). Anything else is run as the command, e.g.
# `vllm serve /model ...` for Dory (serve.sh) or `bash` for debugging.
case "${1:-}" in
  ""|-*) exec /app/start.sh "$@" ;;
  *) exec "$@" ;;
esac
