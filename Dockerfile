ARG MLDEV_VERSION=cu124-py311

FROM nyumerics/ml:${MLDEV_VERSION}

ADD . /tmp/code
RUN --mount=type=cache,target=/root/.cache/pip \
    pushd /tmp/code && \
    micromamba-run pip install --no-cache-dir . && \
    micromamba-run pip uninstall -y llm-calibration && \
    rm -r /tmp/code && \
    popd
