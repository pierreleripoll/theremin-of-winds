/**
 * @brief BUILD OPTIONS for OpenTheremin V4
 * 
 * This file contains macros that will activate customizable
 * options for the CV/GATE support and debug messages on the final build (firmware).
 * 
 * Disabling unnecessary options will slim down the firmware size and ease the CPU load.
 * 
 * Please refer to each option's description.
 */
#ifndef _BUILD_H
#define _BUILD_H

/*
 * AUDIO_FEEDBACK_MODE
 * 
 * - AUDIO_FEEDBACK_ON
 *   enables short audio feedbacks (beeps) at startup and calibration
 *   along the visual indication of the current mode.
 * 
 * - AUDIO_FEEDBACK_OFF
 *   disables the audio feedback (silent), keeping the visual
 *   indication of the current mode.
 * 
 */
#define AUDIO_FEEDBACK_ON   0
#define AUDIO_FEEDBACK_OFF  1
#define AUDIO_FEEDBACK_MODE AUDIO_FEEDBACK_ON

/*
 * CV_OUTPUT_MODE sets options for the CV/GATE output jack
 *
 * for better audio output quality, set it off as it saves precious resources
 * especially if you don't need CV/GATE outputs at all.
 * If activated the GATE pin will output a pseudo-gate
 * signal for Eurorack synths or standard CV/GATE inputs.
 * 
 * WARNING: THE GATE SIGNAL IS ROUTED TO THE RING PIN
 *          OF THE AUDIO OUTPUT JACK! FOR AUDIO OUTPUT
 *          ALWAYS USE A STEREO-MONO SPLITTER TO AVOID
 *          PULSING NOISES GETTING INTO YOUR AUDIO SYSTEM.
 */
#define CV_OUTPUT_MODE_OFF      0       // disables the CV output - this saves resources - for a slightly better audio quality if not needed at all.
#define CV_OUTPUT_MODE_LOG      1       // uses a logarithmic curve for CV output (1V/Oct for Moog & Roland)
#define CV_OUTPUT_MODE_LINEAR   2       // uses a linear transfer function for CV output (819Hz/V for Korg & Yamaha)
#define CV_OUTPUT_MODE          CV_OUTPUT_MODE_LINEAR

// Set the trigger levels for the Gate signal
// Making both values equal risk the gate signal to bounce, leave at least 4 (hysteresis) between both.
// Set both to 128 to disable the Gate Signal.
#define GATE_ON_THRESHOLD_LEVEL     24 // That's the level which will drive the Gate high when volume increases from lower
#define GATE_OFF_THRESHOLD_LEVEL    16 // That's the level which will drive the Gate low when volume decreases from higher


/**
 * SERIAL_PORT_MODE
 * 
 * This macro selects the serial port (UART) mode
 * for external communication (MIDI)
 * and/or to emit debug messages
 * 
 * - SERIAL_PORT_MODE_OFF
 *   disables completely the use of the serial port
 *   reduces code size and CPU overhead
 * 
 * - SERIAL_PORT_DEBUG
 *   will emit debug information that can be read on
 *   the Serial Monitor (or any serial terminal software)
 *   The serial speed for this mode is set by the
 *   SERIAL_SPEED_DEBUG macro defaulted at 57600 bps
 *   NOTE: the higher the speed, the more load onto the
 *   sound generator (DDS-ISR).
 * 
 * - SERIAL_PORT_MIDI
 *   emits serial MIDI note on/note off messages
 *   this is a BASIC implementation, refer to the MIDI expansion
 *   https://www.gaudi.ch/OpenTheremin/index.php/opentheremin-v4/midi-interface
 * 
 *   NOTE: this does not enable the OpenTheremin as a
 *   standard MIDI compliant device, so you will need
 *   a Serial MIDI converter program such as VirtualMIDI,
 *   Hairless MIDI, or SerialMIDI to hook the OpenTheremin
 *   to your favourite DAW.
 *   The MIDI standard uses 31250 bps as serial clock.
 * 
 * NOTE: during debug procedures it is possible to enable
 * both MIDI + DEBUG messages (at fixed speed of 31250 bps)
 * with a bit-wise OR mask on both constants as follows:
 * #define SERIAL_PORT_MODE SERIAL_PORT_MODE_DEBUG | SERIAL_PORT_MODE_MIDI
 * 
 * TO DISABLE THE SERIAL PORT COMPLETELY, JUST SET
 * #define SERIAL_PORT_MODE SERIAL_PORT_MODE_OFF
 */
#define SERIAL_PORT_MODE_OFF        0x00
#define SERIAL_PORT_MODE_DEBUG      0x01
#define SERIAL_PORT_MODE_MIDI       0x02
// STREAM: continuous ASCII "P<pitch> V<vol> M<mute>\n" of the linearized
// pitch/volume (the theremin-of-winds wind synth reads this instead of MIDI).
#define SERIAL_PORT_MODE_STREAM     0x04
#define SERIAL_PORT_MODE            SERIAL_PORT_MODE_STREAM

#define SERIAL_SPEED_DEBUG          57600 // serial com speed
#define SERIAL_SPEED_MIDI           31250 // standard MIDI speed
#define SERIAL_SPEED_STREAM         115200 // fast link for the value stream
#define STREAM_INTERVAL_MS          5      // ~200 Hz control rate


/**
 * SERIAL_DEBUG_MESSAGES MACROS
 * 
 * The following macros will "disapper" if the serial debug mode
 * is being disabled - this will reduce the code size and CPU load automatically.
 * DO NOT DELETE THE FOLLOWING LINES
 */
#if SERIAL_PORT_MODE & SERIAL_PORT_MODE_DEBUG
    #define DEBUG_PRINT(x)     Serial.print(x)
    #define DEBUG_PRINTLN(x)   Serial.println(x)
#else
    #define DEBUG_PRINT(x)     ((void)0)
    #define DEBUG_PRINTLN(x)   ((void)0)
#endif

#endif // _BUILD_H
