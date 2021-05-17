//
// Created by Yi-Wei Chen on 5/12/21.
//

#include "mxc.h"
#include "utils_faceid.h"
#include "sigmoid_lut.h"

static const q31_t lut[] = SIGMOID_LUT;

q31_t q_div(q31_t a, q31_t b)
{
    /* pre-multiply by the base (Upscale to Q64 so that the result will be in Q32 format) */
    int64_t temp = (int64_t)a << 14;
    /* Rounding: mid values are rounded up (down for negative values). */
    /* OR compare most significant bits i.e. if (((temp >> 63) & 1) == ((b >> 31) & 1)) */
    if ((temp >= 0 && b >= 0) || (temp < 0 && b < 0)) {
        temp += (b >> 1);    /* OR shift 1 bit i.e. temp += (b >> 1); */
    } else {
        temp -= (b >> 1);    /* OR shift 1 bit i.e. temp -= (b >> 1); */
    }
    return (q31_t)(temp / b);
}

// saturate to range of int32_t
q31_t sat32(int64_t x)
{
    if (x > 2147483647) return 2147483647;
    else if (x < -2147483648) return -2147483648;
    else return (q31_t)x;
}


q31_t q_mul(q31_t a, q31_t b)
{
    q31_t result;
    int64_t temp;
    int64_t K = (1 << (14 - 1));
    temp = (int64_t)a * (int64_t)b; // result type is operand's type
    // Rounding; mid values are rounded up
    temp += K;
    // Correct by dividing by base and saturate result
    result = sat32(temp >> 14);

    return result;
}

/**
 * @addtogroup Sigmoid
 * @{
 */

/**
 * @brief Q17.14 fixed point sigmoid function, returns Q17.14
 * @param[in]       vec_in      pointer to input vector
 * @param[in]       dim_vec     input vector dimension
 * @param[out]      p_out       pointer to output vector
 * @return none.
 *
 * @details
 *
 *  Here, we use 256-entry lookup table and interpolation to implement sigmoid function
 *  min_x = -8 (-131072), max_x = 7.9375 (130048)
 *  min_y = 0.000335350137902423 (5), max_y = 0.999642968177795 (16378)
 *
 */

q31_t sigmoid(q31_t in){
    q31_t upper = 130048;
    q31_t lower = -131072;
    q31_t out;
    q31_t y1, y2, slope, diff;
    uint8_t idx;
    if (in >= upper)
        return lut[255];
    else if (in <= lower)
        return lut[0];
    else {
        idx = (in >> 10) + 128;
        y1 = lut[idx];
        y2 = lut[idx + 1];
        slope = q_div(y2 - y1, 1024);
        diff = in & 1023;
        out = y1 + q_mul(slope, diff);
        return out;
    }
}

void sigmoid_q17p14_q17p14(const q31_t * vec_in, const uint16_t dim_vec, q31_t * p_out)
{
    uint16_t i;

    for (i = 0; i < dim_vec; i++)
    {
        p_out[i] = sigmoid(vec_in[i]);
    }
}