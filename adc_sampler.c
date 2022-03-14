/*
adc_sampler.c
Public Domain
January 2018, Kristoffer KjÃ¦rnes & Asgeir BjÃ¸rgan
Based on example code from the pigpio library by Joan @ raspi forum and github
https://github.com/joan2937 | http://abyz.me.uk/rpi/pigpio/

Compile with:
gcc -Wall -lpthread -o adc_sampler adc_sampler.c -lpigpio -lm

Run with:
sudo ./adc_sampler

This code bit bangs SPI on several devices using DMA.

Using DMA to bit bang allows for two advantages
1) the time of the SPI transaction can be guaranteed to within a
   microsecond or so.

2) multiple devices of the same type can be read or written
   simultaneously.

This code reads several MCP3201 ADCs in parallel, and writes the data to a binary file.
Each MCP3201 shares the SPI clock and slave select lines but has
a unique MISO line. The MOSI line is not in use, since the MCP3201 is single
channel ADC without need for any input to initiate sampling.
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <pigpio.h>
#include <math.h>

/////// USER SHOULD MAKE SURE THESE DEFINES CORRESPOND TO THEIR SETUP ///////
#define ADCS 1      // Number of connected MCP3201.

#define OUTPUT_DATA argv[2] // path and filename to dump buffered ADC data

/* RPi PIN ASSIGNMENTS */
#define MISO1 12    // ADC 1 MISO (BCM 4 aka GPIO 4).


#define MOSI 19     // GPIO for SPI MOSI (BCM 10 aka GPIO 10 aka SPI_MOSI). MOSI not in use here due to single ch. ADCs, but must be defined anyway.
#define SPI_SS 24   // GPIO for slave select (BCM 8 aka GPIO 8 aka SPI_CE0).
#define CLK 23      // GPIO for SPI clock (BCM 11 aka GPIO 11 aka SPI_CLK).
/* END RPi PIN ASSIGNMENTS */

#define BITS 12            // Bits per sample.
#define BX 4               // Bit position of data bit B11. (3 first are t_sample + null bit)
#define B0 (BX + BITS - 1) // Bit position of data bit B0.

#define NUM_SAMPLES_IN_BUFFER 300 // Generally make this buffer as large as possible in order to cope with reschedule.

#define REPEAT_MICROS 32 // Reading every x microseconds. Must be no less than 2xB0 defined above

#define DEFAULT_NUM_SAMPLES 31250 // Default number of samples for printing in the example. Should give 1sec of data at Tp=32us.

int MISO[ADCS]={MISO1}; // Must be updated if you change number of ADCs/MISOs above
/////// END USER SHOULD MAKE SURE THESE DEFINES CORRESPOND TO THEIR SETUP ///////

/**
 * This function extracts the MISO bits for each ADC and
 * collates them into a reading per ADC.    
 *
 * \param adcs Number of attached ADCs
 * \param MISO The GPIO connected to the ADCs data out
 * \param bytes Bytes between readings
 * \param bits Bits per reading
 * \param buf Output buffer
*/
void getReading(int adcs, int *MISO, int OOL, int bytes, int bits, char *buf)
{
   int p = OOL;
   int i, a;

   for (i=0; i < bits; i++) {
      uint32_t level = rawWaveGetOut(p);
      for (a=0; a < adcs; a++) {
         putBitInBytes(i, buf+(bytes*a), level & (1<<MISO[a]));
      }
      p--;
   }
}


int main(int argc, char *argv[])
{
    // Parse command line arguments
    long num_samples = 0;
    if (argc <= 1) {
        fprintf(stderr, "Usage: %s NUM_SAMPLES\n\n"
                        "Example: %s %d\n", argv[0], argv[0], DEFAULT_NUM_SAMPLES);
        exit(1);
    }
    sscanf(argv[1], "%ld", &num_samples);

    // Array over sampled values, into which data will be saved
    uint16_t *val = (uint16_t*)malloc(sizeof(uint16_t)*num_samples*ADCS);

    // SPI transfer settings, time resolution 1us (1MHz system clock is used)
    rawSPI_t rawSPI =
    {
       .clk     =  CLK,  // Defined before
       .mosi    =  MOSI, // Defined before
       .ss_pol  =  1,   // Slave select resting level.
       .ss_us   =  1,   // Wait 1 micro after asserting slave select.
       .clk_pol =  0,   // Clock resting level.
       .clk_pha =  0,   // 0 sample on first edge, 1 sample on second edge.
       .clk_us  =  1,   // 2 clocks needed per bit so 500 kbps.
    };

    // Change timer to use PWM clock instead of PCM clock. Default is PCM
    // clock, but playing sound on the system (e.g. espeak at boot) will start
    // sound systems that will take over the PCM timer and make adc_sampler.c
    // sample at far lower samplerates than what we desire.
    // Changing to PWM should fix this problem.
    gpioCfgClock(5, 0, 0);

    // Initialize the pigpio library
    if (gpioInitialise() < 0) {
       return 1;
    }

    // Set the selected CLK, MOSI and SPI_SS pins as output pins
    gpioSetMode(rawSPI.clk,  PI_OUTPUT);
    gpioSetMode(rawSPI.mosi, PI_OUTPUT);
    gpioSetMode(SPI_SS,      PI_OUTPUT);

    // Flush any old unused wave data.
    gpioWaveAddNew();

    // Construct bit-banged SPI reads. Each ADC reading is stored separatedly
    // along a buffer of DMA commands (control blocks). When the DMA engine
    // reaches the end of the buffer, it restarts on the start of the buffer
    int offset = 0;
    int i;
    char buf[2];
    for (i=0; i < NUM_SAMPLES_IN_BUFFER; i++) {
        buf[0] = 0xC0; // Start bit, single ended, channel 0.

        rawWaveAddSPI(&rawSPI, offset, SPI_SS, buf, 2, BX, B0, B0);
        offset += REPEAT_MICROS;
    }

    // Force the same delay after the last command in the buffer
    gpioPulse_t final[2];
    final[0].gpioOn = 0;
    final[0].gpioOff = 0;
    final[0].usDelay = offset;

    final[1].gpioOn = 0; // Need a dummy to force the final delay.
    final[1].gpioOff = 0;
    final[1].usDelay = 0;

    gpioWaveAddGeneric(2, final);

    // Construct the wave from added data.
    int wid = gpioWaveCreate();
    if (wid < 0) {
        fprintf(stderr, "Can't create wave, buffer size %d too large?\n", NUM_SAMPLES_IN_BUFFER);
        return 1;
    }

    // Obtain addresses for the top and bottom control blocks (CB) in the DMA
    // output buffer. As the wave is being transmitted, the current CB will be
    // between botCB and topCB inclusive.
    rawWaveInfo_t rwi = rawWaveInfo(wid);
    int botCB = rwi.botCB;
    int topOOL = rwi.topOOL;
    float cbs_per_reading = (float)rwi.numCB / (float)NUM_SAMPLES_IN_BUFFER;

    float expected_sample_freq_khz = 1000.0/(1.0*REPEAT_MICROS);

    printf("# Starting sampling: %ld samples (expected Tp = %d us, expected Fs = %.3f kHz).\n",
    num_samples,REPEAT_MICROS,expected_sample_freq_khz);

    // Start DMA engine and start sending ADC reading commands
    gpioWaveTxSend(wid, PI_WAVE_MODE_REPEAT);

    // Read back the samples
    double start_time = time_time();
    int reading = 0;
    int sample = 0;

    while (sample < num_samples) {
        // Get position along DMA control block buffer corresponding to the current output command.
        int cb = rawWaveCB() - botCB;
        int now_reading = (float) cb / cbs_per_reading;

        while ((now_reading != reading) && (sample < num_samples)) {
            // Read samples from DMA input buffer up until the current output command

            // OOL are allocated from the top down. There are BITS bits for each ADC
            // reading and NUM_SAMPLES_IN_BUFFER ADC readings. The readings will be
            // stored in topOOL - 1 to topOOL - (BITS * NUM_SAMPLES_IN_BUFFER).
            // Position of each reading's OOL are calculated relative to the wave's top
            // OOL.
            int reading_address = topOOL - ((reading % NUM_SAMPLES_IN_BUFFER)*BITS) - 1;

            char rx[8];
            getReading(ADCS, MISO, reading_address, 2, BITS, rx);

            // Convert and save to output array
            for (i=0; i < ADCS; i++) {
                val[sample*ADCS+i] = (rx[i*2]<<4) + (rx[(i*2)+1]>>4);
            }

            ++sample;

            if (++reading >= NUM_SAMPLES_IN_BUFFER) {
                reading = 0;
            }
        }
        usleep(1000);
    }

    double end_time = time_time();

    double nominal_period_us = 1.0*(end_time-start_time)/(1.0*num_samples)*1.0e06;
    double nominal_sample_freq_khz = 1000.0/nominal_period_us;

    printf("# %ld samples in %.6f seconds (actual T_p = %f us, nominal Fs = %.2f kHz).\n",
        num_samples, end_time-start_time, nominal_period_us, nominal_sample_freq_khz);

    double output_nominal_period_us = floor(nominal_period_us); //the clock is accurate only to us resolution

    // Path to your data directory/file from previous define
    const char *output_filename = OUTPUT_DATA;

    // Write sample period and data to file
    FILE *adc_data_file = fopen(output_filename, "wb+");
    if (adc_data_file == NULL) {
        fprintf(stderr, "# Couldn't open file for writing: %s (did you remember to change OUTPUT_DATA?)\n", output_filename);
	return 1;
    }

    fwrite(&output_nominal_period_us, sizeof(double), 1, adc_data_file);
    fwrite(val, sizeof(uint16_t), ADCS*num_samples, adc_data_file);
    fclose(adc_data_file);
    printf("# Data written to file. Program ended successfully.\n\n");

    gpioTerminate();
    free(val);

    return 0;
}