#!/usr/bin/env bash
#
# rtp_jpeg_view.sh
#   1. Set the interface IP to <ip>
#   2. Open the firewall for <port>/udp
#   3. Run gst-launch-1.0 to receive an RTP/JPEG video stream
#   4. On exit (including Ctrl+C or failure), remove the firewall rule
#      and restore the original IP configuration
#
# Usage:
#   ./rtp_jpeg_view.sh [IP] [PORT]
#   ./rtp_jpeg_view.sh                     # defaults: 192.168.1.100 5000
#   ./rtp_jpeg_view.sh 192.168.1.50 6000
#
# Environment variables:
#   IFACE=eth0    Network interface (default: the one used by the default route)
#   PREFIX=24     Netmask prefix length for the new IP (default: 24)
#   KEEP_OLD=1    Keep the original IP and add the new one as a secondary
#                 address (recommended over SSH so the session stays alive)
#
# Note: if the system is managed by NetworkManager, replugging the cable or
# restarting the network service may override these settings.

set -euo pipefail

NEW_IP="${1:-192.168.1.100}"
PORT="${2:-5000}"
PREFIX="${PREFIX:-24}"
KEEP_OLD="${KEEP_OLD:-0}"

info()  { printf '\033[1;34m[INFO]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2; }
die()   { printf '\033[1;31m[ERR ]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------- Validate arguments ----------
[[ "$NEW_IP" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || die "Invalid IP address: $NEW_IP"
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT > 0 && PORT < 65536 )) || die "Invalid port: $PORT"

for cmd in ip gst-launch-1.0; do
    command -v "$cmd" >/dev/null || die "Command not found: $cmd"
done
HAS_UFW=0
command -v ufw >/dev/null && HAS_UFW=1 || warn "ufw not found, skipping firewall steps"

# Acquire sudo up front so the password prompt does not appear mid-run
sudo -v || die "sudo privileges are required"

IFACE="${IFACE:-$(ip -4 route show default | awk '{print $5; exit}')}"
[[ -n "$IFACE" ]] || die "Could not detect an interface, set IFACE=<name> manually"
ip link show "$IFACE" >/dev/null 2>&1 || die "No such interface: $IFACE"

# ---------- Save the original state ----------
mapfile -t ORIG_ADDRS < <(ip -4 -o addr show dev "$IFACE" | awk '{print $4}')
ORIG_DEFAULT="$(ip -4 route show default dev "$IFACE" | head -n1 || true)"

IP_CHANGED=0
FW_ADDED=0

# ---------- Restore routine ----------
cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    echo
    info "Restoring configuration..."

    if (( FW_ADDED )); then
        info "Removing firewall rule ${PORT}/udp"
        sudo ufw delete allow "${PORT}/udp" >/dev/null 2>&1 \
            || warn "Failed to remove the rule, check manually: sudo ufw status numbered"
    fi

    if (( IP_CHANGED )); then
        info "Restoring IP configuration on $IFACE"
        sudo ip addr del "${NEW_IP}/${PREFIX}" dev "$IFACE" 2>/dev/null || true

        for addr in "${ORIG_ADDRS[@]}"; do
            if ! ip -4 -o addr show dev "$IFACE" | awk '{print $4}' | grep -qx "$addr"; then
                sudo ip addr add "$addr" brd + dev "$IFACE" 2>/dev/null \
                    || warn "Failed to restore address: $addr"
            fi
        done

        if [[ -n "$ORIG_DEFAULT" ]] && ! ip -4 route show default dev "$IFACE" | grep -q .; then
            # shellcheck disable=SC2086
            sudo ip route add $ORIG_DEFAULT 2>/dev/null || warn "Failed to restore the default route"
        fi
    fi

    info "Current addresses on $IFACE: $(ip -4 -o addr show dev "$IFACE" | awk '{print $4}' | paste -sd' ' -)"
    info "Restore complete"
    exit "$rc"
}
trap cleanup EXIT INT TERM

# ---------- 1. Change the IP ----------
info "Interface: $IFACE"
info "Original addresses: ${ORIG_ADDRS[*]:-(none)}"
info "Assigning new address: ${NEW_IP}/${PREFIX}"

if (( ! KEEP_OLD )); then
    for addr in "${ORIG_ADDRS[@]}"; do
        sudo ip addr del "$addr" dev "$IFACE" 2>/dev/null || true
    done
fi
IP_CHANGED=1
sudo ip addr add "${NEW_IP}/${PREFIX}" brd + dev "$IFACE"
sudo ip link set "$IFACE" up
sleep 1

# ---------- 2. Open the firewall ----------
if (( HAS_UFW )); then
    info "Allowing ${PORT}/udp through the firewall"
    sudo ufw allow "${PORT}/udp"
    FW_ADDED=1
fi

# ---------- 3. Receive the stream ----------
info "Receiving RTP/JPEG on port ${PORT}. Press Ctrl+C to stop."
set +e
gst-launch-1.0 -v udpsrc port="${PORT}" \
    ! application/x-rtp,media=video,clock-rate=90000,encoding-name=JPEG,payload=26 \
    ! rtpjpegdepay \
    ! jpegdec \
    ! autovideosink sync=false
GST_RC=$?
set -e

(( GST_RC != 0 )) && warn "gst-launch-1.0 exited with code: $GST_RC"

# ---------- 4. cleanup() handles the restore ----------
exit 0
