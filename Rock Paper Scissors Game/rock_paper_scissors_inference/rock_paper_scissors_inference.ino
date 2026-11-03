#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <Arduino_OV767X.h>

#include "model_pruned_full.h"

const char* CLASSES[] = {"rock", "paper", "scissors"};
#define NUM_CLASSES (sizeof(CLASSES) / sizeof(CLASSES[0]))

#define INPUT_W 32
#define INPUT_H 32

#define CAM_W 176
#define CAM_H 144

tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

constexpr int tensorArenaSize = 40 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

unsigned short cameraBuffer[CAM_W * CAM_H];

void setup() {
  Serial.begin(115200);
  while (!Serial);
  Serial.println("=== Rock Paper Scissors Camera Inference ===");

  if (!Camera.begin(QCIF, RGB565, 1)) {
    Serial.println("ERROR: Camera init failed!");
    while (1);
  }
  Serial.println("[OK] Camera initialized.");

  delay(500);
  for (int i = 0; i < 3; i++) {
    Camera.readFrame(cameraBuffer);
    delay(100);
  }

  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Model schema mismatch!");
    while (1);
  }
  Serial.println("[OK] Model loaded.");

  tflInterpreter = new tflite::MicroInterpreter(
      tflModel, tflOpsResolver, tensorArena, tensorArenaSize);

  TfLiteStatus allocStatus = tflInterpreter->AllocateTensors();
  if (allocStatus != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed.");
    while (1);
  }
  Serial.println("[OK] Tensors allocated.");

  tflInputTensor  = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  Serial.print("Input type: ");   Serial.println(tflInputTensor->type);
  Serial.print("Input bytes: ");  Serial.println(tflInputTensor->bytes);
  Serial.print("Output type: ");  Serial.println(tflOutputTensor->type);
  Serial.print("Output bytes: "); Serial.println(tflOutputTensor->bytes);

  Serial.println("Ready, capturing every 3 seconds...\n");
}

void loop() {
  Camera.readFrame(cameraBuffer);

  bool frameOk = false;
  for (int i = 0; i < 50; i++) {
    if (cameraBuffer[i] != 0) { frameOk = true; break; }
  }

  if (!frameOk) {
    Serial.println("WARNING: Empty frame, skipping inference.");
    delay(3000);
    return;
  }

  Serial.println("[OK] Frame captured.");

  for (int y = 0; y < INPUT_H; y++) {
    for (int x = 0; x < INPUT_W; x++) {

      int srcX = x * CAM_W / INPUT_W;
      int srcY = y * CAM_H / INPUT_H;
      int srcIdx = srcY * CAM_W + srcX;

      uint16_t pixel = cameraBuffer[srcIdx];

      // Correct byte order for OV7670
      pixel = ((pixel & 0xFF) << 8) | ((pixel >> 8) & 0xFF);

      // Extract RGB565 channels
      uint8_t r = (pixel >> 11) & 0x1F;
      uint8_t g = (pixel >> 5)  & 0x3F;
      uint8_t b =  pixel        & 0x1F;

      // Scale to 8-bit accurately
      r = (r * 527 + 23) >> 6;
      g = (g * 259 + 33) >> 6;
      b = (b * 527 + 23) >> 6;

      // Integer grayscale conversion
      uint8_t gray = (uint8_t)((r * 77 + g * 150 + b * 29) >> 8);

      tflInputTensor->data.uint8[y * INPUT_W + x] = gray;
    }
  }

  Serial.println("[OK] Image preprocessed.");

  Serial.println("<IMG>");
  for (int i = 0; i < INPUT_W * INPUT_H; i++) {
    uint8_t v = tflInputTensor->data.uint8[i];
    if (v < 0x10) Serial.print('0');
    Serial.print(v, HEX);
  }
  Serial.println();
  Serial.println("</IMG>");

  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
  if (invokeStatus != kTfLiteOk) {
    Serial.println("ERROR: Invoke() failed.");
    delay(3000);
    return;
  }
  Serial.println("[OK] Inference complete.");

  Serial.println("\n--- Class scores ---");
  int bestClass = 0;
  uint8_t bestScore = 0;

  for (int i = 0; i < NUM_CLASSES; i++) {
    uint8_t score = tflOutputTensor->data.uint8[i];
    Serial.print("  ");
    Serial.print(CLASSES[i]);
    Serial.print(": ");
    Serial.println(score);
    if (score > bestScore) {
      bestScore = score;
      bestClass = i;
    }
  }

  Serial.println();
  Serial.print(">>> Predicted class: ");
  Serial.println(CLASSES[bestClass]);
  Serial.println("-----------------------------\n");

  delay(3000);
}