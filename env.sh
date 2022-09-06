#------------------------------------------------------------------------------
HERE=$(cd $(dirname $(readlink -f ${BASH_SOURCE})) && pwd)

export PYTHONPATH="${HERE}:${PYTHONPATH}"
export PATH="${HERE}/test:${PATH}"
