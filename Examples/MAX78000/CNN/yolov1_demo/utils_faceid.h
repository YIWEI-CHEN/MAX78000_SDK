#ifndef UTILS_FACEID_H_
#define UTILS_FACEID_H_

#include "uart.h"
#include "sampledata.h"

#include <stdint.h>
typedef int32_t q31_t;
typedef int16_t q15_t;

#define DATA_SIZE_IN (56*56*3)
#define CONSOLE_UART 0
#define MXC_UARTn   MXC_UART_GET_UART(CONSOLE_UART)
#define IMG_SIZE 56
#define GRID_SIZE 8
#define NUM_CLASSES 20
#define NUM_CHANNELS 30
#define NUM_GRIDS 7
#define NUM_BOXES 2
#define NUM_OUTPUTS (7*7*30)
#define BOX_DIMENSION 10

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

// Data input: HWC (little data): 3x160x120
static const uint32_t input_0[] = INPUT_0;
extern uint8_t rxBuffer[DATA_SIZE_IN];

int uart_write(uint8_t* data, unsigned int len);

int uart_read(uint8_t* buffer, unsigned int len);

int wait_for_feedback();

void load_input(int8_t mode);

void sigmoid_q17p14_q17p14(const q31_t * vec_in, const uint16_t dim_vec, q31_t * p_out);

q31_t q_div(q31_t a, q31_t b);

q31_t q_mul(q31_t a, q31_t b);

void inline_softmax_q17p14_q15(q31_t * vec_in, const uint16_t start, const uint16_t end);

void NMS_max(q31_t * vec_in, const uint16_t dim_vec, q31_t* max_box);

#endif
