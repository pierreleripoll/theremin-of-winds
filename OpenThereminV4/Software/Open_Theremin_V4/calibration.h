#include <Arduino.h>

#ifndef _CALIBRATION_H_
#define _CALIBRATION_H_

extern int32_t pitchCalibrationBase;
extern int32_t volCalibrationBase;

void calibration_read();
bool calibration_start();

#endif // _CALIBRATION_H_