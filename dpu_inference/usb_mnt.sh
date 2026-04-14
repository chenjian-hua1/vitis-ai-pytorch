#!/bin/bash

MOUNT_POINT="/mnt/usb"
DEVICE_BASE="/dev/sda"
MAX_PART=10   # 最多檢查到 sda10，可自行調整

# 建立掛載點
sudo mkdir -p $MOUNT_POINT

found=0

for i in $(seq 1 $MAX_PART); do
    DEVICE="${DEVICE_BASE}${i}"
    
    echo "Trying to mount $DEVICE ..."

    # 嘗試掛載
    sudo mount $DEVICE $MOUNT_POINT 2>/dev/null

    # 檢查是否掛載成功
    if mount | grep -q "$DEVICE"; then
        echo "Mounted $DEVICE"

        # 檢查是否有內容
        if [ "$(ls -A $MOUNT_POINT)" ]; then
            echo "✅ Found data in $DEVICE"
            found=1
            break
        else
            echo "⚠️ Empty partition, unmounting..."
            sudo umount $MOUNT_POINT
        fi
    else
        echo "❌ Failed to mount $DEVICE"
    fi
done

if [ $found -eq 0 ]; then
    echo "❗ No valid partition with data found."
else
    echo "🎯 Final mounted device: $DEVICE"
fi
