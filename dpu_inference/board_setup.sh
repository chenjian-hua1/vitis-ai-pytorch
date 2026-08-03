#!/bin/bash
#
# deploy_app.sh - Swap the Xilinx accelerator app and bring up the LAN interface
#
# Usage:
#   ./deploy_app.sh [IP_ADDR] [APP_NAME]
#
# Examples:
#   ./deploy_app.sh                          # 192.168.1.111 + dpu_cam
#   ./deploy_app.sh 192.168.1.120            # override IP only
#   ./deploy_app.sh 192.168.1.120 my_app     # override both
#

set -uo pipefail

# ---------- defaults ----------
DEFAULT_IP="192.168.1.111"
DEFAULT_APP="dpu_cam"
FW_DIR="/lib/firmware/xilinx"
IFACE="eth0"
NETMASK="255.255.255.0"
TOTAL_STEPS=4

# ---------- colors / message helpers ----------
if [ -t 1 ]; then
    C_STEP=$'\033[1;36m'; C_OK=$'\033[1;32m'; C_WARN=$'\033[1;33m'
    C_ERR=$'\033[1;31m';  C_DIM=$'\033[2m';   C_RST=$'\033[0m'
else
    C_STEP=""; C_OK=""; C_WARN=""; C_ERR=""; C_DIM=""; C_RST=""
fi

STEP_NO=0
step() {
    STEP_NO=$((STEP_NO + 1))
    echo
    echo "${C_STEP}==> [${STEP_NO}/${TOTAL_STEPS}] $1${C_RST}"
}
ok()   { echo "    ${C_OK}[ OK ]${C_RST} $1"; }
warn() { echo "    ${C_WARN}[WARN]${C_RST} $1"; }
err()  { echo "    ${C_ERR}[FAIL]${C_RST} $1" >&2; }
run()  { echo "    ${C_DIM}\$ $*${C_RST}"; "$@"; }

die() { err "$1"; echo; exit "${2:-1}"; }

usage() {
    cat <<EOF

deploy_app.sh - Swap the Xilinx accelerator app and bring up the LAN interface

Usage:
  $(basename "$0") [IP_ADDR] [APP_NAME]

Arguments:
  IP_ADDR    IP address for $IFACE          (default: $DEFAULT_IP)
  APP_NAME   accelerator app to load        (default: $DEFAULT_APP)

Examples:
  $(basename "$0")                        # $DEFAULT_IP + $DEFAULT_APP
  $(basename "$0") 192.168.1.120          # override IP only
  $(basename "$0") 192.168.1.120 my_app   # override both

EOF
    exit 0
}

list_apps() {
    if [ -d "$FW_DIR" ]; then
        echo "    ${C_DIM}Available apps in $FW_DIR:${C_RST}"
        ls -1 "$FW_DIR" 2>/dev/null | sed 's/^/      - /' || echo "      (unable to read directory)"
    else
        warn "Directory not found: $FW_DIR"
    fi
}

# ---------- parse arguments ----------
case "${1:-}" in
    -h|--help) usage ;;
esac

IP_ADDR="${1:-$DEFAULT_IP}"
APP_NAME="${2:-}"

echo "${C_STEP}===== Xilinx App Deployment =====${C_RST}"

if [ -z "${1:-}" ]; then
    warn "No IP given, using default: $DEFAULT_IP"
fi

if [ -z "$APP_NAME" ]; then
    APP_NAME="$DEFAULT_APP"
    warn "No app given, using default: $DEFAULT_APP"
    list_apps
fi

# basic IP sanity check
if ! [[ "$IP_ADDR" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
    die "Invalid IP address format: $IP_ADDR"
fi

echo
echo "  Target app : $APP_NAME"
echo "  Target IP  : $IP_ADDR / $NETMASK ($IFACE)"

# ---------- 1. verify the app exists ----------
step "Checking that the app exists in $FW_DIR"
if [ -e "$FW_DIR/$APP_NAME" ]; then
    ok "Found $APP_NAME"
else
    err "$APP_NAME not found in $FW_DIR"
    list_apps
    exit 1
fi

# ---------- 2. unload the current app ----------
step "Unloading the current app (xmutil unloadapp)"
if run sudo xmutil unloadapp; then
    ok "Unloaded"
else
    warn "unloadapp returned non-zero (likely nothing was loaded), continuing"
fi

# ---------- 3. load the new app ----------
step "Loading app: $APP_NAME (xmutil loadapp)"
if run sudo xmutil loadapp "$APP_NAME"; then
    ok "Loaded $APP_NAME"
else
    die "loadapp failed - check the app name and its device-tree overlay" 2
fi

# ---------- 4. configure the LAN interface ----------
step "Configuring $IFACE with $IP_ADDR"
if run sudo ifconfig "$IFACE" "$IP_ADDR" netmask "$NETMASK" up; then
    ok "Network configured"
else
    die "ifconfig failed - check that $IFACE exists (try: ip link show)" 3
fi

echo
echo "    Current state of $IFACE:"
ip addr show "$IFACE" 2>/dev/null | sed 's/^/      /' \
    || ifconfig "$IFACE" | sed 's/^/      /'

echo
echo "${C_OK}===== Done: app=$APP_NAME, ip=$IP_ADDR =====${C_RST}"
echo