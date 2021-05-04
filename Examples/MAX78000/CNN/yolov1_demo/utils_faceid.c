#include "utils_faceid.h"
#include "cnn.h"

uint8_t rxBuffer[DATA_SIZE_IN];


int uart_write(uint8_t* data, unsigned int len)
{
  unsigned int bytes_tx_total = 0;
  unsigned int bytes_tx;

  while(bytes_tx_total < len){
    bytes_tx = MXC_UART_WriteTXFIFO(MXC_UARTn, data + bytes_tx_total, len - bytes_tx_total);
    bytes_tx_total += bytes_tx;
  }

  return 1;
}

int uart_read(uint8_t* buffer, unsigned int len)
{
  unsigned int bytes_rx_total = 0;
  unsigned int bytes_rx;

  while (bytes_rx_total < len){
    bytes_rx = MXC_UART_ReadRXFIFO(MXC_UARTn, buffer + bytes_rx_total, len - bytes_rx_total);
    bytes_rx_total += bytes_rx;
  }

  return 1;
}

int wait_for_feedback(){
  volatile uint8_t read_byte = 0;

  while (1){
    if (MXC_UART_GetRXFIFOAvailable(MXC_UARTn) > 0) {
      read_byte = MXC_UART_ReadCharacter(MXC_UARTn);
      if (read_byte == 100 )
        break;
      else if (read_byte == 200 )
        return 0;
    }          
  }

  return 1;
}

void load_input(int8_t mode)
{
  uint32_t i;
  i = 0;
  uint32_t *in0 = input_0;
  uint32_t number;
  while (i < 56*56*3) {
    number = ((uint32_t)rxBuffer[i]<<16) | ((uint32_t)rxBuffer[i+1]<<8) | ((uint32_t)rxBuffer[i+2]);

    *in0++ = number;
    i += 3;
  }

  // 3-channel 56x56 data input (9408 bytes total / 3136 bytes per channel):
  // HWC 56x56, channels 0 to 2
  memcpy32((uint32_t *) 0x50400000, in0, 56*56);

}