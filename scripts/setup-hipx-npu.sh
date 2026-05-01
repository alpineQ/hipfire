#!/usr/bin/env bash
# Bring up the AMD XDNA NPU stack on a Strix Halo (gfx1151 + aie2p) box.
#
# Why this script exists: the Linux 7.0 mainline `amdxdna` driver
# (Ubuntu in-tree) does not implement hugepage-backed BO mmap that the
# AIE-2P firmware requires for `MSG_OP_MAP_HOST_BUFFER`. The OOT
# `amd/xdna-driver` (DKMS) does. This script installs that, plus the
# user-side memlock config the runtime needs.
#
# After running, reboot or `newgrp` and:
#   - `xrt-smi examine` should enumerate the device
#   - `target/debug/xdna-probe` (this crate) prints AIE metadata
#   - `target/debug/xdna-hwctx-roundtrip` shows 40/40 winning combos

set -euo pipefail

USER_NAME="${SUDO_USER:-${USER}}"
REPO_DIR="${HOME}/xdna-driver"

if [[ "${EUID}" -eq 0 ]]; then
    echo "Don't run this as root — it'll prompt for sudo where needed." >&2
    exit 1
fi

echo "==> Installing build prerequisites"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    dkms build-essential cmake git pkg-config libelf-dev jq \
    "linux-headers-$(uname -r)" \
    libxrt2 libxrt-npu2 libxrt-utils-npu libxrt-utils

echo "==> Cloning amd/xdna-driver (recursive)"
if [[ ! -d "${REPO_DIR}" ]]; then
    git clone --recursive https://github.com/amd/xdna-driver.git "${REPO_DIR}"
else
    git -C "${REPO_DIR}" pull --recurse-submodules
fi

echo "==> Building xrt_plugin-amdxdna"
( cd "${REPO_DIR}/build" && ./build.sh -release )

DEB_PATH=$(ls "${REPO_DIR}/build/Release/"xrt_plugin*-amdxdna.deb | head -1)
echo "==> Installing ${DEB_PATH##*/}"
# --force-depends because the deb wants xrt-base 2.23 but Ubuntu ships
# 2.21; the kernel-module portion (which is what we actually need) is
# unaffected by the userspace XRT version.
sudo dpkg -i --force-depends "${DEB_PATH}"

echo "==> Reloading amdxdna module"
sudo modprobe -r amdxdna || true
sudo modprobe amdxdna
sleep 1

echo "==> Configuring memlock for ${USER_NAME}"
LIMITS_FILE="/etc/security/limits.d/90-${USER_NAME}-memlock.conf"
sudo tee "${LIMITS_FILE}" >/dev/null <<EOF
${USER_NAME} hard memlock unlimited
${USER_NAME} soft memlock unlimited
EOF

echo
echo "==> Done. Verify:"
echo "    modinfo amdxdna | grep version    # expect '1.0.0' or higher"
echo "    sudo dmesg | grep PASID            # expect 'PASID address mode enabled'"
echo "    xrt-smi examine                    # expect device enumeration"
echo "    ulimit -l                          # expect 'unlimited' after re-login"
echo
echo "If 'ulimit -l' still shows a number, log out and back in (or 'newgrp -' refresh)."
