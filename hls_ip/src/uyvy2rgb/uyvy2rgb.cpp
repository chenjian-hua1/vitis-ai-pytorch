#include <ap_int.h>
#include <stdio.h>

/* Sub Module
out = in - 128;
out[sign] = !in[MSB]; // >=128
out[sign-1:0] = !in[MSB-1:0]; // value
*/
void subtract_128(ap_uint<8> &in, ap_int<8> &out) {
    // 正: in[6:0]   負(2補數表示法): 128 - (-(in[6:0]-128)) = in[6:0]
    out.range(6,0) = in.range(6,0);
    // 正負號取取決 in>=128 (看第7位元上面位元)
    out[7] = !in[7];

    // Debug Code
    std::cout << out << std::endl;
}




/* Top Module
Max Input : 4096x4096 (2^12)

uyuy_axi_bus : uyuy image data pointer (read)
rgb_axi_bus : rgb image data pointer (write)
img_w , img_h : image width , height
*/
void uyvy2rgb(ap_uint<128> *uyvy_axi_bus, ap_uint<128> *rgb_axi_bus, ap_uint<12> img_w, ap_uint<12> img_h) {

}

int main() {
    std::cout << "application start" << std::endl;
    ap_uint<8> in = 150;
    ap_int<8> out;

    subtract_128(in, out);

    return 0;
}