/*
 *  Open.Theremin control software for Arduino UNO
 *  Version 4.5
 *  Copyright (C) 2010-2025 by Urs Gaudenz
 *
 *  Open.Theremin control software is free software: you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License as published
 *  by the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Open.Theremin control software is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along with
 *  the Open.Theremin control software. If not, see <http://www.gnu.org/licenses/>.
 *
 *  With important contributions by 
 *  David Harvey
 *  Michael Margolis
 *  "Theremingenieur" Thierry Frenkel
 */
 
/**
Building the code
=================
build.h contains #defines that control the compilation of the code
please refer to the comments in the actual `build.h` for detailed description.

           
Structure of the code
=====================
** Open_Theremin_V4.ino **
This file. Creates and hooks up the application object to the standard
Arduino setup() and loop() methods.

** application.h/application.cpp **
Main application code. Holds the state of the app (playing, calibrating), deals
with initialisation and the app main loop, reads pitch and volume changed flags
from the interrupt handlers and sets pitch and volume values which the timer
interrupt sends to the DAC.

** calibration.h/calibration.cpp **
Self-calibration routines

** SPImcpDAC.h **
Internal library for communicating with the MCP492x DAC chips (audio, calibration and CV outputs).

** build.h **
Preprocessor definitions for customizable build options (see above).

** hw.h **
Definitions for hardware button and LED.

** ihandlers.h/ihandlers.cpp
Interrupt handler code and volatile variables implementing actual DDS (Direct Digital Synthesis) wave generator.

** wavetable_<N>.c **
Wavetable data for a variety of sounds as 1024 points array. Switchable via the Timbre potentiometer.

** timer.h/timer.cpp **
Definitions and functions for setting delays in ticks and in milliseconds

** EEPROM.h
Interface library for persistance storage of the calibrated values of the VCOs via the calibration DACs.

*/

#include "application.h"


void setup() {
  theremin_setup();
}

void loop() {
  theremin_loop();
}
