#include <Adafruit_NeoPixel.h>
#ifdef __AVR__
#include <avr/power.h>
#endif

#define SOLENOID_PIN       2
#define HOPPER_PIN       3
#define TOP_CONVEYOR_PIN       A4
#define BOTTOM_CONVEYOR_PIN       A5
#define LED_PIN            6
//LED_PIN8 is just because I did not want to solder...
#define LED_PIN8           5


// How many NeoPixels are attached to the Arduino?
#define NUMPIXELS      95

Adafruit_NeoPixel pixels = Adafruit_NeoPixel(NUMPIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

int old_pixel = 0;

void setup() {
  
  // This is for Trinket 5V 16MHz, you can remove these three lines if you are not using a Trinket
#if defined (__AVR_ATtiny85__)
  if (F_CPU == 16000000) clock_prescale_set(clock_div_1);
#endif
  // End of trinket special code
  Serial.begin(115200);
  pixels.begin(); // This initializes the NeoPixel library.
  pixels.setPixelColor(0, pixels.Color(255,255,255));
  pixels.show();
}

void loop() {
  if (Serial.available() > 0) {
    int input = Serial.parseInt();
    if (input < NUMPIXELS) {
      //for(int i=0;i < NUMPIXELS;i++){
      pixels.setPixelColor(old_pixel , pixels.Color(0,0,0));
      pixels.show();
      pixels.setPixelColor(old_pixel  + 47, pixels.Color(0,0,0));
      pixels.show();
      for(int i=0;i < 6;i++){
        pixels.setPixelColor(input+i, pixels.Color(255,255,255));
        pixels.show();
        pixels.setPixelColor(input+47 +i, pixels.Color(255,255,255));
        pixels.show();
      }
      old_pixel = input;
    } 
    if (input == 100) {
      for(int i=0;i < 47;i++){
      pixels.setPixelColor(i, pixels.Color(255,255,255));
      pixels.show();
      }
    } 
   
    if (input == 101){
      for(int i=0;i < NUMPIXELS;i++){
      pixels.setPixelColor(i, pixels.Color(0,0,0));
      pixels.show();
      }
    } 
    
    if (Serial.read() == '\n') {};
  }
}

