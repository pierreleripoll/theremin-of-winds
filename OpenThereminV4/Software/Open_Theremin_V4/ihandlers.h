#include <Arduino.h>

#ifndef _IHANDLERS_H
#define _IHANDLERS_H

extern volatile uint16_t pitch;                 // Pitch value
extern volatile uint16_t vol;                   // Volume value
extern volatile uint16_t vScaledVolume;         // Volume byte
extern volatile int16_t pitchCV;                // Pitch CV value
extern volatile uint16_t volCV;                 // Volume CV value

extern volatile uint16_t pitch_counter;         // Pitch counter
extern volatile uint16_t pitch_counter_l;       // Last value of pitch counter

extern volatile uint16_t vol_counter;           // Pitch counter
extern volatile uint16_t vol_counter_l;         // Last value of pitch counter

extern volatile uint16_t timer_overflow_counter;         // counter for frequency measurement

extern volatile bool volumeValueAvailable;      // Volume read flag
extern volatile bool pitchValueAvailable;       // Pitch read flag
extern volatile bool volumeCVAvailable;         // Volume CV flag
extern volatile bool pitchCVAvailable;          // Pitch CV flag

extern volatile uint8_t  vWavetableSelector;
extern volatile uint16_t vPointerIncrement;     // Table pointer increment

// the number of wavetable being loaded in the DDS generator (ihandlers.cpp)
extern const uint8_t num_wavetables;

inline void setWavetableSampleAdvance(uint16_t val) { vPointerIncrement = val;}

void ihInitialiseTimer();
void ihInitialiseInterrupts();
void ihInitialisePitchMeasurement();
void ihInitialiseVolumeMeasurement();

#endif // _IHANDLERS_H
