#!/bin/bash
#
# pc_stream_recv_mac.sh  (macOS version)
#   1. Add <ip> as an alias (secondary) address on the interface
#   2. If the macOS Application Firewall is on, allow gst-launch-1.0
#   3. Run gst-launch-1.0 to receive an RTP/JPEG video stream
#   4. On exit (including Ctrl+C or failure), remove the alias address
#      and the firewall entry that this script added
#
# The original address is never touched, so this is safe over SSH and
# does not fight with macOS's configd / DHCP.
#
# Usage:
#   ./pc_stream_recv_mac.sh [IP] [PORT]
#   ./pc_stream_recv_mac.sh                    # defaults: 192.168.1.100 5000
#   ./pc_stream_recv_mac.sh 192.168.1.50 6000
#
# Environment variables:
#   IFACE=en0               Network interface (default: the one used by the default route)
#   PREFIX=24               Netmask prefix length for the new IP (default: 24)
#   VIDEOSINK=glimagesink   Video sink (default: autovideosink)
#
# Requirements:
#   brew install gstreamer
#
# Compatible with the stock macOS bash 3.2.

set -euo pipefail

NEW_IP="${1:-192.168.1.100}"
PORT="${2:-5000}"
PREFIX="${PREFIX:-24}"
VIDEOSINK="${VIDEOSINK:-autovideosink}"
FW=/usr/libexec/ApplicationFirewall/socketfilterfw

info()  { printf '\033[1;34m[INFO]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2; }
die()   { printf '\033[1;31m[ERR ]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------- Helpers ----------
valid_ip() {
    local a b c d o
    [[ "$1" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || return 1
    IFS=. read -r a b c d <<<"$1"
    for o in "$a" "$b" "$c" "$d"; do
        (( 10#$o <= 255 )) || return 1
    done
}

ip2int() {
    local a b c d
    IFS=. read -r a b c d <<<"$1"
    echo $(( (10#$a << 24) | (10#$b << 16) | (10#$c << 8) | 10#$d ))
}

int2ip() {
    echo "$(( ($1 >> 24) & 255 )).$(( ($1 >> 16) & 255 )).$(( ($1 >> 8) & 255 )).$(( $1 & 255 ))"
}

# Resolve symlinks (Homebrew installs gst-launch-1.0 as a symlink);
# the Application Firewall needs the real binary path.
resolve_path() {
    local p="$1" t
    while [[ -L "$p" ]]; do
        t="$(readlink "$p")"
        if [[ "$t" == /* ]]; then p="$t"; else p="$(dirname "$p")/$t"; fi
    done
    echo "$(cd "$(dirname "$p")" && pwd -P)/$(basename "$p")"
}

iface_addrs() {
    ifconfig "$IFACE" | awk '/inet /{print $2}'
}

# ---------- Validate environment and arguments ----------
[[ "$(uname -s)" == "Darwin" ]] || die "This script is for macOS only"

valid_ip "$NEW_IP" || die "Invalid IP address: $NEW_IP"
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT > 0 && PORT < 65536 )) || die "Invalid port: $PORT"
[[ "$PREFIX" =~ ^[0-9]+$ ]] && (( PREFIX >= 1 && PREFIX <= 32 )) || die "Invalid prefix: $PREFIX"

for cmd in ifconfig route gst-launch-1.0; do
    command -v "$cmd" >/dev/null || die "Command not found: $cmd (try: brew install gstreamer)"
done

if command -v gst-inspect-1.0 >/dev/null; then
    for el in udpsrc rtpjpegdepay jpegdec videoconvert "$VIDEOSINK"; do
        gst-inspect-1.0 "$el" >/dev/null 2>&1 \
            || die "GStreamer element not found: $el (try: brew reinstall gstreamer)"
    done
fi

# Acquire sudo up front so the password prompt does not appear mid-run
sudo -v || die "sudo privileges are required"

IFACE="${IFACE:-$(route -n get default 2>/dev/null | awk '/interface:/{print $2; exit}')}"
[[ -n "$IFACE" ]] || die "Could not detect an interface, set IFACE=<name> manually (e.g. IFACE=en0)"
ifconfig "$IFACE" >/dev/null 2>&1 || die "No such interface: $IFACE"

# ---------- Work out the alias netmask ----------
# macOS recommends a /32 netmask for an alias that falls inside a subnet
# the interface already has; otherwise duplicate subnet routes can occur.
NEW_INT="$(ip2int "$NEW_IP")"
if (( PREFIX == 32 )); then
    MASK_INT=$(( 0xFFFFFFFF ))
else
    MASK_INT=$(( (0xFFFFFFFF << (32 - PREFIX)) & 0xFFFFFFFF ))
fi
ALIAS_MASK="$(int2ip "$MASK_INT")"

SAME_SUBNET=0
while read -r addr mask; do
    [[ -n "${addr:-}" && -n "${mask:-}" ]] || continue
    m=$(( mask ))
    if (( (NEW_INT & m) == ($(ip2int "$addr") & m) )); then
        SAME_SUBNET=1
    fi
done <<<"$(ifconfig "$IFACE" | awk '/inet /{print $2, $4}')"

if (( SAME_SUBNET )); then
    ALIAS_MASK="255.255.255.255"
fi

ALIAS_ADDED=0
FW_APP_ADDED=0
GST_BIN="$(resolve_path "$(command -v gst-launch-1.0)")"

# ---------- Restore routine ----------
cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    echo
    info "Restoring configuration..."

    if (( FW_APP_ADDED )); then
        info "Removing $GST_BIN from the Application Firewall"
        sudo "$FW" --remove "$GST_BIN" >/dev/null 2>&1 \
            || warn "Failed to remove the firewall entry, check System Settings > Network > Firewall"
    fi

    if (( ALIAS_ADDED )); then
        info "Removing alias $NEW_IP from $IFACE"
        sudo ifconfig "$IFACE" -alias "$NEW_IP" 2>/dev/null \
            || warn "Failed to remove alias, run manually: sudo ifconfig $IFACE -alias $NEW_IP"
    fi

    info "Current addresses on $IFACE: $(iface_addrs | paste -sd' ' -)"
    info "Restore complete"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---------- 1. Add the alias address ----------
info "Interface: $IFACE"
info "Original addresses: $(iface_addrs | paste -sd' ' -)"

if iface_addrs | grep -qxF "$NEW_IP"; then
    warn "$NEW_IP is already on $IFACE, leaving it as is (it will not be removed on exit)"
else
    (( SAME_SUBNET )) && info "New IP is in an existing subnet on $IFACE, using netmask 255.255.255.255"
    info "Adding alias: $NEW_IP netmask $ALIAS_MASK"
    sudo ifconfig "$IFACE" alias "$NEW_IP" netmask "$ALIAS_MASK"
    ALIAS_ADDED=1
    sleep 1
fi

# ---------- 2. Application Firewall ----------
if [[ -x "$FW" ]] && "$FW" --getglobalstate 2>/dev/null | grep -qi "enabled"; then
    if "$FW" --getblockall 2>/dev/null | grep -qi "enabled"; then
        warn "Firewall is set to block ALL incoming connections; the stream will not arrive."
        warn "Turn off 'Block all incoming connections' in System Settings > Network > Firewall."
    fi

    if "$FW" --listapps 2>/dev/null | grep -qF "$GST_BIN"; then
        info "gst-launch-1.0 is already in the firewall list, leaving it as is"
    else
        info "Allowing $GST_BIN through the Application Firewall"
        sudo "$FW" --add "$GST_BIN" >/dev/null
        FW_APP_ADDED=1
        sudo "$FW" --unblockapp "$GST_BIN" >/dev/null
    fi
else
    info "Application Firewall is off, skipping firewall steps"
fi

# ---------- 3. Receive the stream ----------
info "Receiving RTP/JPEG on port ${PORT} (sink: ${VIDEOSINK}). Press Ctrl+C to stop."
set +e
gst-launch-1.0 -v udpsrc port="${PORT}" \
    ! application/x-rtp,media=video,clock-rate=90000,encoding-name=JPEG,payload=26 \
    ! rtpjpegdepay \
    ! jpegdec \
    ! videoconvert \
    ! "${VIDEOSINK}" sync=false
GST_RC=$?
set -e

(( GST_RC != 0 )) && warn "gst-launch-1.0 exited with code: $GST_RC"

# ---------- 4. cleanup() handles the restore ----------
exit 0