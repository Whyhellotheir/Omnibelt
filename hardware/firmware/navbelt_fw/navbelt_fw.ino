/*
  NavBelt ESP32 firmware  —  ESP32-WROOM-32 dev board -> ULN2803APG -> coin motors
  ---------------------------------------------------------------------------------
  Arduino IDE 2: Boards Manager -> install "esp32 by Espressif" -> Board: "ESP32 Dev Module"
  Serial: 115200 baud, one command per line (newline-terminated). Every command replies OK / ERR / data.

  COMMANDS
    PING               -> PONG
    TEST               -> buzzes M1..M8 one at a time (wiring check)
    M <1-8> <0-100> S  -> motor steady at that strength      e.g.  M 3 70 S
    M <1-8> <0-100> P4 -> motor pulsing at 4 Hz              e.g.  M 1 80 P6
    M <1-8> 0 S        -> that motor off
    ALL 0              -> everything off
    SWEEP              -> "STOP" pattern: fast pulse around the belt twice
    ARRIVE             -> "ARRIVED" pattern: two pulses on the front three motors
    WD 0  /  WD 1      -> turn the 500 ms safety watchdog off (for typing tests by hand) / on

  SAFETY: strength capped at 65 % (3 V motors on ~4 V), max 3 motors on at once,
          and if no command arrives for 500 ms all motors stop (watchdog).
*/
#include <Arduino.h>

const int NUM = 8;
// M1 front, M2 front-right, M3 right, M4 back-right, M5 back-left, M6 left, M7 front-left, M8 ring
// All on the SAME side of the ESP32 board (the side you leave uncovered on the breadboard).
const int PINS[NUM] = {13, 14, 33, 25, 26, 27, 32, 12};

const int   PWM_FREQ    = 20000;   // above hearing range
const int   PWM_BITS    = 8;       // 0..255
const float DUTY_CAP    = 0.65;    // 100 % command = 65 % real
const int   MAX_ON      = 3;
const unsigned long WATCHDOG_MS = 500;

struct Motor { int duty; float hz; bool on; };
Motor m[NUM];

enum Seq { NONE, SWEEP, ARRIVE, TESTSEQ };
Seq seq = NONE;
unsigned long seqStart = 0, lastCmd = 0;
int testLast = -1;
bool wdEnabled = true;
String buf;

void pwmAttach(int i) {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(PINS[i], PWM_FREQ, PWM_BITS);
#else
  ledcSetup(i, PWM_FREQ, PWM_BITS);
  ledcAttachPin(PINS[i], i);
#endif
}

void pwmWrite(int i, int v) {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(PINS[i], v);
#else
  ledcWrite(i, v);
#endif
}

int raw(int pct) { return (int)(constrain(pct, 0, 100) / 100.0f * DUTY_CAP * 255.0f); }

int activeCount() {
  int c = 0;
  for (int i = 0; i < NUM; i++) if (m[i].on) c++;
  return c;
}

void allOff() {
  for (int i = 0; i < NUM; i++) { m[i].duty = 0; m[i].hz = 0; m[i].on = false; pwmWrite(i, 0); }
  seq = NONE;
}

void startSeq(Seq s) { allOff(); seq = s; seqStart = millis(); testLast = -1; }

void handleLine(String s) {
  s.trim();
  if (s.length() == 0) return;
  lastCmd = millis();
  s.toUpperCase();

  if (s == "PING")   { Serial.println("PONG"); return; }
  if (s == "TEST")   { startSeq(TESTSEQ); Serial.println("OK TEST"); return; }
  if (s == "SWEEP")  { startSeq(SWEEP);   Serial.println("OK"); return; }
  if (s == "ARRIVE") { startSeq(ARRIVE);  Serial.println("OK"); return; }
  if (s.startsWith("ALL")) { allOff(); Serial.println("OK"); return; }
  if (s.startsWith("WD")) {
    wdEnabled = !s.endsWith("0");
    Serial.println(wdEnabled ? "OK watchdog ON" : "OK watchdog OFF");
    return;
  }
  if (s.startsWith("M")) {
    int n = 0, d = 0; char pat[8] = "S";
    int got = sscanf(s.c_str(), "M %d %d %7s", &n, &d, pat);
    if (got < 2 || n < 1 || n > NUM) { Serial.println("ERR usage: M <1-8> <0-100> S|P<hz>"); return; }
    int i = n - 1;
    seq = NONE;                                  // direct control cancels patterns
    if (d <= 0) { m[i].on = false; m[i].duty = 0; Serial.println("OK"); return; }
    if (!m[i].on && activeCount() >= MAX_ON) { Serial.println("ERR max 3 motors on"); return; }
    m[i].duty = d;
    m[i].hz   = (pat[0] == 'P') ? atof(pat + 1) : 0.0f;
    m[i].on   = true;
    Serial.println("OK");
    return;
  }
  Serial.println("ERR unknown command");
}

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < NUM; i++) { pwmAttach(i); pwmWrite(i, 0); m[i] = {0, 0, false}; }
  lastCmd = millis();
  Serial.println("NAVBELT READY");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') { handleLine(buf); buf = ""; }
    else if (c != '\r' && buf.length() < 64) buf += c;
  }

  unsigned long now = millis();
  if (seq == NONE && wdEnabled && activeCount() > 0 && now - lastCmd > WATCHDOG_MS) allOff();

  int out[NUM] = {0};
  if (seq == NONE) {
    for (int i = 0; i < NUM; i++) {
      if (!m[i].on) continue;
      bool phaseOn = (m[i].hz <= 0) || (fmodf(now / 1000.0f * m[i].hz, 1.0f) < 0.5f);
      out[i] = phaseOn ? raw(m[i].duty) : 0;
    }
  } else if (seq == SWEEP) {                       // 7 belt motors, 70 ms each, twice
    unsigned long t = now - seqStart; int step = t / 70;
    if (step >= 14) seq = NONE; else out[step % 7] = raw(100);
  } else if (seq == ARRIVE) {                      // front-left, front, front-right: 2 pulses
    unsigned long t = now - seqStart;
    if (t >= 1000) seq = NONE;
    else if ((t % 500) < 300) { out[6] = raw(100); out[0] = raw(100); out[1] = raw(100); }
  } else if (seq == TESTSEQ) {                     // each motor 400 ms on, 200 ms off
    unsigned long t = now - seqStart; int step = t / 600;
    if (step >= NUM) { seq = NONE; Serial.println("TEST DONE"); }
    else {
      if (step != testLast) { Serial.printf("TEST M%d (GPIO%d)\n", step + 1, PINS[step]); testLast = step; }
      if ((t % 600) < 400) out[step] = raw(100);
    }
  }
  for (int i = 0; i < NUM; i++) pwmWrite(i, out[i]);
}
