#include "ihandlers.h"
#include "SPImcpDAC.h"
#include "timer.h"
#include "build.h"
#include "hw.h"

#include "theremin_sintable0.h"
#include "theremin_sintable1.h"
#include "theremin_sintable2.h"
#include "theremin_sintable3.h"
#include "theremin_sintable4.h"
#include "theremin_sintable5.h"
#include "theremin_sintable6.h"
#include "theremin_sintable7.h"

const int16_t *const wavetables[] = {
    sine_table0,
    sine_table1,
    sine_table2,
    sine_table3,
    sine_table4,
    sine_table5,
    sine_table6,
    sine_table7
};
// number of included wavetables to adapt potentiometer hysteresis and range.
const uint8_t num_wavetables = (sizeof(wavetables) / sizeof(wavetables[0]));
#define DDS_WAVETABLE_RESOLUTION 0x3ff      // future expansion from 1024 to 2048 point wavetables

static const uint32_t MCP_DAC_BASE = 2048;

volatile uint16_t vScaledVolume = 0;        // output level amplitude
volatile uint16_t vPointerIncrement = 0;    // phase accumulator increment

volatile uint16_t pitch = 0;                // Pitch value
volatile uint16_t pitch_counter = 0;        // Pitch counter
volatile uint16_t pitch_counter_l = 0;      // Last value of pitch counter

volatile bool volumeValueAvailable = false; // Volume read flag
volatile bool pitchValueAvailable = false;  // Pitch read flag
volatile bool reenableInt1 = false;         // reeanble Int1

volatile uint16_t vol;                      // Volume value
volatile uint16_t vol_counter = 0;          // volume counter
volatile uint16_t vol_counter_i = 0;        // Volume counter Timer 1
volatile uint16_t vol_counter_l;            // Last value of volume counter

volatile int16_t pitchCV;                   // Pitch CV value
volatile uint16_t volCV;                    // Volume CV value
volatile bool volumeCVAvailable = false;    // Volume CV flag
volatile bool pitchCVAvailable = false;     // Pitch CV flag

volatile uint16_t timer_overflow_counter;   // counter for frequency measurement
volatile uint8_t vWavetableSelector = 0;    // wavetable selector

static volatile uint16_t pointer = 0;       // Table pointer 16-bit: 10 bits integer + 6 bits fraction

static volatile uint8_t debounce_p, debounce_v = 0; // Counters for debouncing


/**
 * @brief Initializes hardware timers used for pitch frequency measurement and system timing.
 *
 * This function sets up:
 * - Timer1 in normal mode with input capture on rising edge at full 16 MHz clock rate,
 *     used to measure the frequency of the pitch oscillator via the ICP1 pin (PD5 VO_PITCH).
 * - Timer0 with Arduino default Fast PWM mode and a 64× prescaler,
 *     which supports core timing functions like `millis()` and `delay()`.
 *
 * Timer1 Input Capture Interrupt is enabled to allow precise measurement of pitch events.
 * Timer0 Overflow Interrupt is also enabled, to expand the resolution.
 *
 * @note
 * - Timer1 is configured to run without prescaling for maximum frequency resolution.
 * - The ICES1 bit ensures rising edge detection on ICP1 (PD5) VO_PITCH.
 * - No waveform output is enabled; timers are used strictly for counting.
 */
void ihInitialiseTimer() {
    // --- Timer1: Pitch frequency measurement (ICP1 on PD5) VO_PITCH ---        
    /**
     * TCCR1A controls waveform generation mode and output compare behavior.
     * Setting it to 0 disables waveform output on OC1A/OC1B and enables normal counting mode.
     */
    TCCR1A = 0; // Normal mode, no PWM output
    
    /**
     * TCCR1B:
     * - ICES1: Input Capture Edge Select → 1 = capture on rising edge
     * - CS10:    Clock Select → 1 = no prescaler (Timer1 counts at full 16 MHz)
     */
    TCCR1B = (1 << ICES1) | (1 << CS10); // Rising edge capture, no prescaling

    /**
     * TIMSK1:
     * - ICIE1: Enable Timer1 Input Capture interrupt (TIMER1_CAPT_vect)
     */
    TIMSK1 = (1 << ICIE1); // Enable input capture interrupt

    // --- Timer0: Arduino core timebase (used by millis/delay) ---

    /**
     * TCCR0A:
     * - WGM01 + WGM00 = 1 → Fast PWM mode (Arduino default)
     */
    TCCR0A = 0x03; // Fast PWM mode

    /**
     * TCCR0B:
     * - CS01 + CS00 = 1 → Prescaler = clk/64 (Timer0 runs at 250 kHz)
     */
    TCCR0B = 0x03; // Clock /64 prescaler

    /**
     * TIMSK0:
     * - TOIE0: Enable Timer0 Overflow Interrupt (used by Arduino core)
     */
    TIMSK0 = 0x01; // Enable Timer0 overflow interrupt
}

/**
 * @brief Configures external interrupts for wave generator and volume measurement.
 *
 * This function sets up the External Interrupt Control Register (EICRA) to trigger interrupts
 * on rising edges for both INT0 (volume gate, F_VOL) and INT1 (sample clock / wave generator tick).
 * These are connected respectively to PD2 (INT0) and PD3 (INT1) on the ATmega328P.
 *
 * - INT0 is used to latch volume oscillator timing information from the flip-flop gate.
 * - INT1 triggers the main waveform ISR for audio output and pitch/volume tracking.
 *
 * External interrupts are enabled in the mask register (EIMSK), and a software flag
 * (`reenableInt1`) is set to allow re-arming INT1 from within the ISR after it's temporarily disabled.
 *
 * @note This setup is AVR-specific using direct register manipolation.
 */
void ihInitialiseInterrupts() {
    // --- Configure interrupt trigger type for INT0 (PD2 F_VOL) and INT1 (PD3 SAMPLE_CLK) ---

    /* EICRA – External Interrupt Control Register A
            ISC00 + ISC01 = 1 + 1 = Rising edge trigger for INT0 (F_VOL) PD2
            ISC10 + ISC11 = 1 + 1 = Rising edge trigger for INT1 (SAMPLE_CLK ISR) PD3
    */
    EICRA = (1 << ISC00) | (1 << ISC01) | (1 << ISC11) | (1 << ISC10);

    reenableInt1 = true; // Internal flag to allow INT1 to be re-enabled in the main ISR

    /* EIMSK – External Interrupt Mask Register
            Enables INT0 (bit 0) and INT1 (bit 1)
    */
    EIMSK = (1 << INT0) | (1 << INT1);
}

/**
 * @brief Prepares Timer1 for direct pitch oscillator frequency measurement.
 *
 * This routine disables external interrupts and configures Timer1 to count freely,
 * using the full 16 MHz system clock, with overflow interrupt enabled.
 * It’s typically called during absolute pitch calibration.
 *
 * The RF oscillator signal (VO_PITCH) enters on PD5 (ICP1 / T1), and its rising edge
 * transitions are captured by periodically reading the counter value (TCNT1).
 *
 * Configuration:
 * - Disables external INT0 and INT1 (EIMSK = 0) for F_VOL and SAMPLE_CLK
 * - Timer1 normal mode with no waveform generation (TCCR1A = 0) counter only
 * - Enables Timer1 Overflow Interrupt (used to measure longer periods)
 *
 * @note No prescaling is used for maximum resolution. The overflow ISR (TIMER1_OVF_vect)
 *       increments a software counter to extend timing range beyond 16 bits.
 */
void ihInitialisePitchMeasurement() {
    reenableInt1 = false;
    EIMSK = 0;                      // Disable External Interrupts INT0 and INT1
    TCCR1A = 0;                     // Timer1 Normal Mode, no PWM or compare output
    TIMSK1 = (1 << TOIE1);          // Enable Timer1 Overflow Interrupt (TIMER1_OVF_vect)
}

/**
 * @brief Configures Timer0 and Timer1 to perform volume oscillator measurement.
 *
 * This mode is used during calibration or diagnostics to capture the absolute frequency
 * of the volume RF oscillator (VO_VOL) using internal timers only.
 *
 * Configuration steps:
 * - Disables external interrupts (EIMSK = 0)
 * - Timer0 is used in Normal mode (TCCR0A = 0) and set to trigger compare-match interrupts.
 * - Timer1 is used as a free-running counter with prescaler 1024 to extend timing range.
 *
 * Volume signal transitions (VO_VOL) are detected by checking TCNT1 in relation
 * to Timer0 Compare Match A (OCR0A), which generates periodic samples.
 *
 * @note This is distinct from regular volume gating via F_VOL/INT0 and is more suited
 *       for precise, absolute measurements when direct oscillator observation is required.
 */
void ihInitialiseVolumeMeasurement() {
    reenableInt1 = false;

    EIMSK = 0;                      // Disable External Interrupts INT0 and INT1
    TIMSK1 = 0;                     // Disable Timer1 Input Capture and Overflow interrupts

    TCCR0A = 0;                     // Timer0 Normal mode (no PWM)
    TIMSK0 = (1 << OCIE0A);         // Enable Timer0 Compare Match A interrupt (TIMER0_COMPA_vect)
    OCR0A = 0xff;                   // Set Compare Match A value to max (255)

    TCCR1A = 0;                     // Timer1 Normal mode
    TCCR1B = (1 << CS10) | (1 << CS12); // Prescaler = 1024 (clock = 15.625 kHz)
    TCCR1C = 0;                     // No forced compare
}

/**
 * @brief Main DDS waveform interrupt handler triggered by SAMPLE_CLK (INT1).
 *
 * This ISR is invoked at the audio sample rate (31.25 kHz) and performs:
 * - DAC waveform update and pointer increment.
 * - Rising-edge detection of F_PITCH and F_VOL signals.
 * - Frequency capture using Timer1.
 * - CV output if enabled.
 *
 * ## Signal Input Explanation
 * - `F_PITCH_STATE` reads pin PB0 (digital pin 8) — the F_PITCH signal after flip-flop.
 *     Defined as: `(PINB & (1 << PORTB0))` → true when PB0 is HIGH.
 *
 * - `F_VOL_STATE` reads pin PD2 (digital pin 2) — the F_VOL signal after flip-flop.
 *     Defined as: `(PIND & (1 << PORTD2))` → true when PD2 is HIGH.
 *
 * These signals are fed from the logic divider+flip-flop gates for pitch and volume.
 * A rising edge indicates a new gated RF cycle suitable for timing measurement.
 *
 * - debounce method: 3-sample debounce followed by 2-sample confirmation.
 *
 * Externaly generated 31250 Hz Interrupt for WAVE generator (32us) 
 * 16MHz / 512 (2^9) = 31250 SAMPLE_CLK PD3
 * 
 * @note Keep interrupt duration below sample period (~32 µs).
 *       This interrupt vector has already the highest priority so
 *       there is no need to enable/disable interrupts within this method.
 *       therefore, all calls to EIMSK, sei() or cli() have been removed.
 */
ISR(INT1_vect) {
#ifdef TH_DEBUG
    HW_LED_RED_ON;
#endif

    SPImcpDAClatch();                           // Latch previous DAC value before sending a new sample

    // extract the fractional offset from the 10.6 fixed-point phase accumulator (pointer).
    // Shifting by 6 matches the 64-step sub-sample precision — efficient and avoids floating-point.
    uint16_t offset = (pointer >> 6) & DDS_WAVETABLE_RESOLUTION; // 10-bit table index
    // prepare the new sample for output
    int16_t waveSample = (int16_t)pgm_read_word_near(wavetables[vWavetableSelector] + offset);
    uint32_t scaledSample = ((int32_t)waveSample * (uint32_t)vScaledVolume) >> 16;
    SPImcpDACsend(scaledSample + MCP_DAC_BASE);             // Send result to audio DAC (5.5 us)
    pointer += vPointerIncrement;                           // Advance wavetable phase pointer
    incrementTimer();  // update 32us timer

    // PB0 == F_PITCH
    if (F_PITCH_PIN) { debounce_p++; }
    if (debounce_p == 3) {
        //cli();
        pitch_counter = ICR1;                               // Get Timer-Counter 1 value
        pitch = pitch_counter - pitch_counter_l;            // Counter change since last interrupt -> pitch value
        pitch_counter_l = pitch_counter;                    // Set actual value as new last value
    } else if (debounce_p == 5) { pitchValueAvailable = true; }

    // PD2 == F_VOL
    if (F_VOL_PIN) { debounce_v++; }
    if (debounce_v == 3) {
        //cli();
        vol_counter = vol_counter_i;                        // Get Timer-Counter 1 value
        vol = (vol_counter - vol_counter_l);                // Counter change since last interrupt
        vol_counter_l = vol_counter;                        // Set actual value as new last value
    } else if (debounce_v == 5) { volumeValueAvailable = true; }
    
    #if CV_OUTPUT_MODE != CV_OUTPUT_MODE_OFF
        if (pitchCVAvailable) {
            /* 
            * At the very end to limit potential interference with sound generation
            * Priority on pitchCV, volumeCV if occurring at the same moment might be delayed by 32us without harm,
            * but it's better to do max. 1 additional SPI transaction per interrupt do leave enough headroom for the loop.
            */
            SPImcpDAC3Asend(pitchCV);                       // Send result to DAC (pitchCV out) (5.5 us)
            pitchCVAvailable = false;                       // Reset flag
        } else if (volumeCVAvailable) {
            SPImcpDAC3Bsend(volCV);                         // Send result to DAC (volCV out) (5.5 us)
            volumeCVAvailable = false;                      // Reset flag
        }
    #endif

#ifdef TH_DEBUG
    HW_LED_RED_OFF;
#endif
}

/* VOLUME read - interrupt service routine for capturing volume counter value */
ISR(INT0_vect) {
    vol_counter_i = TCNT1;
    debounce_v = 0;
}

/* PITCH read - interrupt service routine for capturing pitch counter value */
ISR(TIMER1_CAPT_vect) {
    debounce_p = 0;
}

/* PITCH read absolute frequency - interrupt service routine for calibration measurement */
ISR(TIMER0_COMPA_vect) {
    timer_overflow_counter++;
}

/* VOLUME read absolute frequency - interrupt service routine for calibration measurement */
ISR(TIMER1_OVF_vect) {
    timer_overflow_counter++;
}
