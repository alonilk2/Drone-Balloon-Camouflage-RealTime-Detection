 //Face Tracker using OpenCV and Arduino
//by Shubham Santosh

#include<Servo.h>
//
Servo servoVer; //Vertical Servo
Servo servoHor; //Horizontal Servo
int x;
int y;
int prevX;
int prevY;
void setup()
{
  Serial.begin(9600);
  servoVer.attach(5); //Attach Vertical Servo to Pin 5
  servoHor.attach(11); //Attach Horizontal Servo to Pin 6
  servoVer.write(90);
  servoHor.write(90);
}
void Pos()
{
  if(prevX != x || prevY != y)
  {
    int servoX = map(x, 1280, 0, 70, 179);
    int servoY = map(y, 1024, 0, 179, 95);
    servoX = min(servoX, 179);
    servoX = max(servoX, 70);
    servoY = min(servoY, 179);
    servoY = max(servoY, 95);
    
    servoHor.write(servoX);
    servoVer.write(servoY);
  }
}
void loop()
{
  if(Serial.available() > 0)
  {
    if(Serial.read() == 'X')
    {
      x = Serial.parseInt();
      if(Serial.read() == 'Y')
      {
        y = Serial.parseInt();
       Pos();
      }
    }
    while(Serial.available() > 0)
    {
      Serial.read();
    }
  }
}



//Servo x, y;
//int width = 1280, height = 1024;  // total resolution of the video
//int xpos = 90, ypos = 90;  // initial positions of both Servos
//
//void setup() {
//
//  Serial.begin(115200);
//  x.attach(9);
//  y.attach(10);
//  // Serial.print(width);
//  //Serial.print("\t");
//  //Serial.println(height);
//  x.write(xpos);
//  y.write(ypos);
//}
//const int angle = 2;   // degree of increment or decrement
//
//void loop() {
//  if (Serial.available() > 0)
//  {
//    int x_mid, y_mid;
//    if (Serial.read() == 'X')
//    {
//      x_mid = Serial.parseInt();  // read center x-coordinate
//      if (Serial.read() == 'Y')
//        y_mid = Serial.parseInt(); // read center y-coordinate
//    }
//    /* adjust the servo within the squared region if the coordinates
//        is outside it
//    */
//    if (x_mid > width / 2 + 30)
//      xpos += angle;
//    if (x_mid < width / 2 - 30)
//      xpos -= angle;
//    if (y_mid < height / 2 + 30)
//      ypos -= angle;
//    if (y_mid > height / 2 - 30)
//      ypos += angle;
//
//
//    // if the servo degree is outside its range
//    if (xpos >= 180)
//      xpos = 180;
//    else if (xpos <= 0)
//      xpos = 0;
//    if (ypos >= 180)
//      ypos = 180;
//    else if (ypos <= 0)
//      ypos = 0;
//
//    x.write(xpos);
//    y.write(ypos);
//
//  }
//}
