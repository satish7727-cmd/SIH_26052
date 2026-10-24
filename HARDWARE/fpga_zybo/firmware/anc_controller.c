#include <stdio.h>
#include "xil_io.h"
#include "xparameters.h"

#define REG_MU_STEP_ADDR       0x43C00000
#define REG_MODE_PROTECT_ADDR  0x43C00004
#define REG_FAST_IMPULSE_ADDR  0x43C00008

void update_anc_controller(uint32_t raw_spi_packet) {
    uint8_t speech_status   = (raw_spi_packet >> 24) & 0xFF;
    uint8_t noise_type      = (raw_spi_packet >> 16) & 0xFF;
    uint8_t impulse_status  = (raw_spi_packet >> 8)  & 0xFF;
    
    uint32_t mu_value = 0x00000400; // Default step size
    uint32_t mode_flag = 0x01;       // Default normal ANC

    // Fast Impulse or Explosive Shockwave detected
    if (impulse_status == 1 || noise_type == 0x0A) {
        mu_value = 0x00000010; // Freeze/Clamp adaptive step-size to prevent divergence
        mode_flag = 0x02;      // Protection / Mute Mode
    } else if (speech_status == 1) {
        mu_value = 0x00000200; // Optimal step-size for speech presence
        mode_flag = 0x01;
    }

    Xil_Out32(REG_MU_STEP_ADDR, mu_value);
    Xil_Out32(REG_MODE_PROTECT_ADDR, mode_flag);
}

int main() {
    printf("[ZYBO Z7-10] ANC Controller Firmware Initialized.\n");
    while(1) {
        uint32_t hw_impulse = Xil_In32(REG_FAST_IMPULSE_ADDR);
        if (hw_impulse) {
            Xil_Out32(REG_MU_STEP_ADDR, 0x00000000); // Immediate Hardware Freeze
        }
    }
    return 0;
}
