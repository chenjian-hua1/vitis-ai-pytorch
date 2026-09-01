// xrt_probe.cpp — 確認這塊板子的 XRT 能不能配 BO 並取得實體位址
//
// 編譯(板子上或 SDK 內皆可):
//   g++ -std=c++17 xrt_probe.cpp -o xrt_probe \
//       -I/usr/include/xrt -lxrt_coreutil -lpthread -luuid
//
// 若 header 不在 /usr/include/xrt,先找:
//   find / -name "xrt_bo.h" 2>/dev/null
//
// 這支程式只做三件事:開裝置、配一塊 BO、印出實體位址。
// 能印出非 0 的位址,就表示 XRT 這條路可行。

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <exception>

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>

int main(int argc, char** argv) {
    const size_t size     = (argc > 1) ? std::strtoul(argv[1], nullptr, 0)
                                       : 8u * 1024 * 1024;
    const int    group_id = (argc > 2) ? std::atoi(argv[2]) : 0;

    printf("嘗試配置 %zu bytes,group_id=%d\n", size, group_id);

    try {
        xrt::device dev(0);
        printf("裝置已開啟\n");

        try {
            printf("  名稱: %s\n", dev.get_info<xrt::info::device::name>().c_str());
        } catch (...) {
            printf("  (取不到裝置名稱,不影響)\n");
        }

        xrt::bo bo(dev, size, xrt::bo::flags::normal, group_id);

        const uint64_t phys = bo.address();
        auto* virt = bo.map<uint8_t*>();

        printf("\nBO 配置成功\n");
        printf("  實體位址 : 0x%llx\n", (unsigned long long)phys);
        printf("  虛擬位址 : %p\n", (void*)virt);
        printf("  大小     : %zu bytes\n", size);

        if (phys == 0) {
            printf("\n[!] 實體位址是 0 —— 這個平台不透過 address() 提供位址,\n"
                   "    XRT 這條路走不通,建議改用 --heap(DMA-BUF Heaps)\n");
            return 1;
        }
        if (phys & 0xFFF) {
            printf("\n[!] 位址沒有對齊 page,不尋常,建議再確認\n");
        }

        // 寫個 pattern 並同步,確認 map 與 sync 都能動
        std::memset(virt, 0xA5, 4096);
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, 4096, 0);
        bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, 4096, 0);
        printf("  map/sync : 正常(前 4 bytes = %02x %02x %02x %02x)\n",
               virt[0], virt[1], virt[2], virt[3]);

        printf("\n可行。接下來可以用:\n");
        printf("  ./rk_app --devmem --xrt %zu --selftest\n", size / (1024 * 1024));
        return 0;

    } catch (const std::exception& e) {
        printf("\n失敗: %s\n", e.what());
        printf("\n常見原因:\n");
        printf("  - 沒有載入 xclbin,zocl 拒絕配置 BO\n");
        printf("      試試: xmutil listapps / xmutil loadapp <app>\n");
        printf("  - group_id 不對,換一個試試(0~3)\n");
        printf("      ./xrt_probe %zu 1\n", size);
        printf("  - 這版 XRT 需要先建立 hardware context\n");
        printf("\n若都不行,DMA-BUF Heaps(--heap)是更省事的選擇。\n");
        return 1;
    }
}