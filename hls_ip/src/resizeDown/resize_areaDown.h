// 計算該數字需要使用多少位元
constexpr int LOG2_CEIL(int x) {
    // 定義域：x >= 1
    // ceil(log2(1)) = 0, ceil(log2(2)) = 1, ceil(log2(3)) = 2, ...
    int r = 0;
    int p = 1;
    // 直到 2^r 超過x
    while (p <= x) {
        p*=2;
        ++r;
    }
    return r;
}

// 2^? 形式其中一個bit=1 其他0
constexpr bool IS_POW2(int x) {
    return x > 0 && ((x & (x - 1)) == 0);
}

// 計算 for index (0..maxium-1) 需要的位元寬度
// 規則：若 maxium 是 2 的冪次方 -> LOG2_CEIL(maxium + 1)
// 否則 -> LOG2_CEIL(maxium)
constexpr int FOR_IDX_BITS(int maxium) {
    return IS_POW2(maxium) ? LOG2_CEIL(maxium + 1) : LOG2_CEIL(maxium);
}