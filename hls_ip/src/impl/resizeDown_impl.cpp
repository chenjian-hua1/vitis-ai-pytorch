/******************************************************************************
 * resize_impl.cpp
 *
 * resize 運算實作，宣告見 resize_impl.h
 *****************************************************************************/

#include "resize_impl.h"

/* ================================================================
 *  DSP 計算：P = (A + D) * B
 *
 *  對應 DSP48E2：前加器 (A + D) + 乘法器 (* B)
 *  INLINE 後每個呼叫點各自展開成一顆 DSP
 *
 *  位元寬：A、D 24 bit unsigned（轉 signed 25 bit <= 27）
 *          B    16 bit unsigned（轉 signed 17 bit <= 18）
 *  lane 不溢位由呼叫端的值域保證（見 dsp_shared 註解）
 * ================================================================ */
ap_uint<PROD_W> dsp_addmul(ap_uint<OPW> a, ap_uint<OPW> d, ap_uint<SRW> b)
{
#pragma HLS INLINE
    ap_uint<OPW>    pre = a + d;       /* 前加器 */
    ap_uint<PROD_W> m   = pre * b;     /* 乘法器 */
#pragma HLS BIND_OP variable=m op=mul impl=dsp
    return m;
}


/* ================================================================
 *  兩組共用一顆 DSP
 *
 *  pack = 1（2 倍）：一顆 DSP 服務兩組
 *    A = {a_hi, a_lo}，D = {d_hi, d_lo}      每 lane 12 bit
 *    P = (A + D) * B
 *    p_hi = P[23:12]，p_lo = P[11:0]
 *    成立條件：每 lane (a + d) <= 1020 < 2^10，B <= 3
 *              -> 前加不進位、乘積 <= 3060 < 2^12 不溢出 lane
 *
 *  pack = 0（3 倍）：一顆 DSP 只服務一組（B 為 16 bit，乘積 28 bit，
 *    塞不進 12-bit lane，無法打包）
 *    A = a_lo，D = d_lo，p_lo = P（完整乘積），p_hi = 0
 *
 *  拆出來之後，每一組都拿到「自己的乘積」，後續取位不必再管 lane：
 *    輸出 8 bit = p[FRAC+7 : FRAC]，寫回 = p[11:0]
 * ================================================================ */
void dsp_shared(bool             pack,
                       ap_uint<ACCW>    a_hi,
                       ap_uint<ACCW>    d_hi,
                       ap_uint<ACCW>    a_lo,
                       ap_uint<ACCW>    d_lo,
                       ap_uint<SRW>     b,
                       ap_uint<PROD_W> &p_hi,
                       ap_uint<PROD_W> &p_lo)
{
#pragma HLS INLINE
    ap_uint<OPW> A2 = (a_hi, a_lo);            /* 打包：{hi, lo} */
    ap_uint<OPW> D2 = (d_hi, d_lo);
    ap_uint<OPW> A  = pack ? A2 : (ap_uint<OPW>)a_lo;
    ap_uint<OPW> D  = pack ? D2 : (ap_uint<OPW>)d_lo;

    ap_uint<PROD_W> P = dsp_addmul(A, D, b);   /* 唯一的 DSP 呼叫點 */

    if (pack) {
        p_hi = P.range(2*ACCW - 1, ACCW);
        p_lo = P.range(ACCW - 1, 0);
    } else {
        p_hi = 0;
        p_lo = P;
    }
}

/* ================================================================
 *  resize PE：一顆 DSP 份的運算
 *
 *    h0 = u0 + u1，h1 = u2 + v            （兩種模式共用這兩個加法器）
 *    3 倍：只用 lo 組，a = h0 (p0+p1)，d = h1 (p2+line_buf)
 *    2 倍：hi 組 = 較左欄，a = h0 (p0+p1)，d = line_buf[i]
 *          lo 組 = 較右欄，a = h1 (p2+p3)，d = line_buf[i+1]
 *
 *  值域：3 倍 h0<=510、h1<=1785；2 倍 h0,h1<=510、lane 和 <=1020
 * ================================================================ */
void resize_pe(ap_uint<8>       u0,
               ap_uint<8>       u1,
               ap_uint<8>       u2,
               ap_uint<ACCW>    v,
               ap_uint<OPW>     lbp,
               bool             s3,
               ap_uint<SRW>     mul_b,
               ap_uint<PROD_W> &p_hi,
               ap_uint<PROD_W> &p_lo)
{
#pragma HLS INLINE
    ap_uint<ACCW> h0    = u0 + u1;
    ap_uint<ACCW> h1    = u2 + v;
    ap_uint<ACCW> lb_hi = lbp.range(OPW  - 1, ACCW);
    ap_uint<ACCW> lb_lo = lbp.range(ACCW - 1, 0);

    ap_uint<ACCW> a_lo  = s3 ? h0 : h1;
    ap_uint<ACCW> d_lo  = s3 ? h1 : lb_lo;

    dsp_shared(!s3, h0, lb_hi, a_lo, d_lo, mul_b, p_hi, p_lo);
}
