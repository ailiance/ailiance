#!/usr/bin/env python3
"""Rebuild cpp domain with MICROCONTROLLER / EMBEDDED C++ focus.

Sources (priority order, all verified SPDX):
  A) iamtarun/code_instructions_120k_alpaca  (Apache-2.0)  — embedded MCU filter
  B) bigcode/self-oss-instruct-sc2-exec-filter-50k (Apache-2.0)  — embedded MCU filter
  C) iamtarun/code_instructions_120k_alpaca  (Apache-2.0)  — hardware-flavored C++
  D) OSHWA scraped data  — synthetic embedded C++ instruction pairs
  E) iamtarun/code_instructions_120k_alpaca  (Apache-2.0)  — generic C++ fallback

Target: 60%+ embedded-MCU content, remainder useful general C++ for embedded devs.
Output: 2850 train / 150 valid in data/hf-traced/cpp/

Usage:
    cd ~/ailiance && uv run python scripts/rebuild_cpp_embedded.py
"""
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset

SEED = 42
TARGET = 3000
VALID_RATIO = 0.05
OUT = Path("data/hf-traced")
MANIFEST_PATH = OUT / "MANIFEST_niche.json"
OSHWA_PATH = Path("data/scraped/oshwa/train.jsonl")

# ─── Embedded / MCU keyword sets ────────────────────────────────────────────

MCU_KEYWORDS: list[str] = [
    "GPIO", "UART", "SPI", "I2C", "ADC", "DAC", "PWM", "DMA",
    "interrupt", "ISR", "volatile", "register",
    "PORTA", "PORTB", "DDRB", "TCCR", "OCR", "TIMSK",
]

PLATFORM_KEYWORDS: list[str] = [
    "Arduino", "ATmega", "AVR", "STM32", "HAL_", "ESP32",
    "PIC", "ARM", "Cortex", "CMSIS", "FreeRTOS", "RTOS",
]

HW_FUNC_KEYWORDS: list[str] = [
    "pinMode", "digitalWrite", "analogRead", "Serial.begin",
    "Wire.begin", "attachInterrupt", "noInterrupts", "sei()", "cli()",
]

HARDWARE_COMPONENT_KEYWORDS: list[str] = [
    "sensor", "motor", "LED", "button", "relay", "servo", "stepper",
    "encoder", "thermocouple", "accelerometer", "gyroscope",
    "magnetometer", "display", "LCD", "OLED",
]

ALL_EMBEDDED_KEYWORDS: list[str] = (
    MCU_KEYWORDS + PLATFORM_KEYWORDS + HW_FUNC_KEYWORDS + HARDWARE_COMPONENT_KEYWORDS
)

# Bit manipulation patterns (regex)
BIT_MANIP_PATTERNS: list[re.Pattern] = [
    re.compile(r"0x[0-9A-Fa-f]+"),      # hex literals
    re.compile(r"<<\s*\d"),              # left shift
    re.compile(r"\|\s*="),               # OR-assign
    re.compile(r"&\s*="),               # AND-assign
    re.compile(r"~\s*\("),              # bitwise NOT
]

# ─── C++ basic markers (for generic fallback) ───────────────────────────────

CPP_MARKERS: list[str] = [
    "#include", "std::", "cout", "cin", "int main", "nullptr",
    "template<", "class ", "namespace", "vector<",
]

# ─── Exclusion markers ──────────────────────────────────────────────────────

JAVA_MARKERS: list[str] = [
    "public static void", "System.out", "import java",
    "public class ", "String[]",
]

PYTHON_MARKERS: list[str] = ["def ", "import ", "print("]

CSHARP_MARKERS: list[str] = [
    "using System", "using Windows", "public sealed",
    "IActionResult", "Console.WriteLine",
]

JS_MARKERS: list[str] = [
    "console.log", "document.", "require(", "module.exports",
    "function(", "const ", "let ", "var ",
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def count_embedded_keywords(text: str) -> tuple[int, list[str]]:
    """Count distinct embedded keywords found in text."""
    found: list[str] = []
    for kw in MCU_KEYWORDS + PLATFORM_KEYWORDS + HW_FUNC_KEYWORDS:
        if kw in text:
            found.append(kw)
    # Bit manipulation counts as one keyword if any pattern matches
    bit_found = any(p.search(text) for p in BIT_MANIP_PATTERNS)
    if bit_found:
        found.append("bit_manip")
    return len(found), found


def count_hardware_components(text: str) -> tuple[int, list[str]]:
    """Count hardware component keywords (case-insensitive)."""
    text_lower = text.lower()
    found = [kw for kw in HARDWARE_COMPONENT_KEYWORDS if kw.lower() in text_lower]
    return len(found), found


def count_cpp_markers(text: str) -> tuple[int, list[str]]:
    """Count distinct C++ markers found in text."""
    found = [m for m in CPP_MARKERS if m in text]
    return len(found), found


def is_contaminated(text: str) -> bool:
    """Check for Java/Python/C#/JS contamination."""
    if sum(1 for m in JAVA_MARKERS if m in text) >= 2:
        return True
    if sum(1 for m in CSHARP_MARKERS if m in text) >= 2:
        return True
    if sum(1 for m in JS_MARKERS if m in text) >= 3:
        return True
    # Python contamination only if no C++ context
    py_count = sum(1 for m in PYTHON_MARKERS if m in text)
    cpp_count, _ = count_cpp_markers(text)
    if py_count >= 2 and cpp_count < 2:
        return True
    return False


def make_msg(user: str, assistant: str, provenance: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ],
        "_provenance": provenance,
    }


def extract_alpaca_text(row: dict) -> tuple[str, str]:
    """Extract user/assistant text from alpaca-format row."""
    instruction = row.get("instruction", "") or ""
    inp = row.get("input", "") or ""
    output = row.get("output", "") or ""
    user_text = instruction.strip()
    if inp.strip():
        user_text = f"{user_text}\n\n{inp.strip()}"
    return user_text, output.strip()


# ─── Source A: Embedded MCU from iamtarun ────────────────────────────────────

def load_source_a(ds) -> tuple[list[dict], dict]:
    """Filter iamtarun for EMBEDDED C++ (>=2 MCU/platform/hw-func keywords)."""
    source_id = "iamtarun/code_instructions_120k_alpaca"
    license_ = "Apache-2.0"
    print(f"\n{'='*60}")
    print(f"Source A: {source_id} — embedded MCU filter")
    print(f"{'='*60}")

    records: list[dict] = []
    stats = {"total": len(ds), "pass": 0, "contaminated": 0}

    for idx, row in enumerate(ds):
        user_text, output = extract_alpaca_text(row)
        if not user_text or not output:
            continue

        combined = f"{user_text}\n{output}"

        if is_contaminated(combined):
            stats["contaminated"] += 1
            continue

        emb_count, _ = count_embedded_keywords(combined)
        if emb_count < 2:
            continue

        stats["pass"] += 1
        provenance = {
            "source": source_id,
            "license": license_,
            "record_id": str(idx),
            "domain_tag": "embedded-mcu",
        }
        records.append(make_msg(user_text, output, provenance))

    print(f"  Embedded MCU pass (>=2 kw): {stats['pass']}")
    print(f"  Contaminated rejected: {stats['contaminated']}")
    return records, stats


# ─── Source B: Embedded MCU from bigcode ─────────────────────────────────────

def load_source_b() -> tuple[list[dict], dict]:
    """Filter bigcode/self-oss-instruct for embedded C++."""
    source_id = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
    license_ = "Apache-2.0"
    print(f"\n{'='*60}")
    print(f"Source B: {source_id} — embedded MCU filter")
    print(f"{'='*60}")

    ds = load_dataset(source_id, split="train")
    print(f"  Raw records: {len(ds)}")

    records: list[dict] = []
    stats = {"total": len(ds), "pass": 0, "contaminated": 0}

    for idx, row in enumerate(ds):
        # bigcode uses 'instruction' and 'output' or 'response'
        instruction = (
            row.get("instruction", "")
            or row.get("problem", "")
            or row.get("prompt", "")
            or ""
        )
        output = (
            row.get("output", "")
            or row.get("response", "")
            or row.get("solution", "")
            or ""
        )

        if not instruction.strip() or not output.strip():
            continue

        combined = f"{instruction}\n{output}"

        if is_contaminated(combined):
            stats["contaminated"] += 1
            continue

        emb_count, _ = count_embedded_keywords(combined)
        if emb_count < 2:
            continue

        stats["pass"] += 1
        provenance = {
            "source": source_id,
            "license": license_,
            "record_id": str(idx),
            "domain_tag": "embedded-mcu",
        }
        records.append(make_msg(instruction.strip(), output.strip(), provenance))

    print(f"  Embedded MCU pass (>=2 kw): {stats['pass']}")
    print(f"  Contaminated rejected: {stats['contaminated']}")
    return records, stats


# ─── Source C: Hardware-flavored C++ from iamtarun ───────────────────────────

def load_source_c(ds, exclude_ids: set[str]) -> tuple[list[dict], dict]:
    """C++ records with #include AND hardware component keywords."""
    source_id = "iamtarun/code_instructions_120k_alpaca"
    license_ = "Apache-2.0"
    print(f"\n{'='*60}")
    print(f"Source C: {source_id} — hardware-flavored C++")
    print(f"{'='*60}")

    records: list[dict] = []
    stats = {"total": len(ds), "pass": 0, "contaminated": 0, "already_in_a": 0}

    for idx, row in enumerate(ds):
        if str(idx) in exclude_ids:
            stats["already_in_a"] += 1
            continue

        user_text, output = extract_alpaca_text(row)
        if not user_text or not output:
            continue

        combined = f"{user_text}\n{output}"

        if is_contaminated(combined):
            stats["contaminated"] += 1
            continue

        # Must have #include
        if "#include" not in combined:
            continue

        # Must have at least one hardware component keyword
        hw_count, _ = count_hardware_components(combined)
        if hw_count < 1:
            continue

        stats["pass"] += 1
        provenance = {
            "source": source_id,
            "license": license_,
            "record_id": str(idx),
            "domain_tag": "embedded-mcu",
        }
        records.append(make_msg(user_text, output, provenance))

    print(f"  Hardware C++ pass: {stats['pass']}")
    print(f"  Already in Source A: {stats['already_in_a']}")
    print(f"  Contaminated rejected: {stats['contaminated']}")
    return records, stats


# ─── Source D: Synthetic from OSHWA ──────────────────────────────────────────

# Templates for synthetic embedded C++ instruction pairs
SYNTHETIC_TEMPLATES: list[dict] = [
    {
        "pattern": "gpio_setup",
        "instruction": "Write embedded C++ code to configure {pin} as {direction} on {platform} and {action} an {component}.",
        "code": """\
#include <{header}>

{defines}

void setup() {{
    // Configure {pin} as {direction}
    {setup_code}
}}

void loop() {{
    {loop_code}
    {delay}
}}""",
        "variants": [
            {
                "pin": "GPIO pin 13", "direction": "output", "platform": "Arduino",
                "action": "blink", "component": "LED",
                "header": "Arduino.h",
                "defines": "const int LED_PIN = 13;",
                "setup_code": "pinMode(LED_PIN, OUTPUT);",
                "loop_code": "    digitalWrite(LED_PIN, HIGH);\n    delay(500);\n    digitalWrite(LED_PIN, LOW);",
                "delay": "    delay(500);",
            },
            {
                "pin": "PA5", "direction": "output", "platform": "STM32 HAL",
                "action": "toggle", "component": "LED",
                "header": "stm32f4xx_hal.h",
                "defines": "#define LED_PIN GPIO_PIN_5\n#define LED_PORT GPIOA",
                "setup_code": "HAL_GPIO_Init(LED_PORT, &GPIO_InitStruct);",
                "loop_code": "    HAL_GPIO_TogglePin(LED_PORT, LED_PIN);",
                "delay": "    HAL_Delay(500);",
            },
            {
                "pin": "GPIO2", "direction": "output", "platform": "ESP32",
                "action": "toggle", "component": "LED",
                "header": "Arduino.h",
                "defines": "const int LED_PIN = 2;",
                "setup_code": "pinMode(LED_PIN, OUTPUT);",
                "loop_code": "    digitalWrite(LED_PIN, !digitalRead(LED_PIN));",
                "delay": "    delay(1000);",
            },
            {
                "pin": "PB5", "direction": "output", "platform": "ATmega328P (bare AVR)",
                "action": "blink", "component": "LED",
                "header": "avr/io.h>\n#include <util/delay.h",
                "defines": "",
                "setup_code": "DDRB |= (1 << PB5);  // Set PB5 as output",
                "loop_code": "    PORTB ^= (1 << PB5);  // Toggle PB5",
                "delay": "    _delay_ms(500);",
            },
        ],
    },
    {
        "pattern": "uart_init",
        "instruction": "Write C++ code to initialize UART at {baud} baud on {platform} and send a {message}.",
        "code": """\
#include <{header}>

void setup() {{
    {uart_init}
}}

void loop() {{
    {send_code}
    {delay}
}}""",
        "variants": [
            {
                "baud": "9600", "platform": "Arduino",
                "message": "hello world string",
                "header": "Arduino.h",
                "uart_init": "Serial.begin(9600);",
                "send_code": '    Serial.println("Hello from Arduino!");',
                "delay": "    delay(1000);",
            },
            {
                "baud": "115200", "platform": "ESP32",
                "message": "sensor reading",
                "header": "Arduino.h",
                "uart_init": "Serial.begin(115200);\n    Serial.println(\"ESP32 UART ready\");",
                "send_code": '    int sensorVal = analogRead(34);\n    Serial.printf("Sensor: %d\\n", sensorVal);',
                "delay": "    delay(500);",
            },
            {
                "baud": "9600", "platform": "ATmega328P (bare AVR)",
                "message": "character string",
                "header": "avr/io.h>\n#include <util/delay.h",
                "uart_init": "// Set baud rate to 9600 (16MHz clock)\n    UBRR0H = 0;\n    UBRR0L = 103;\n    // Enable transmitter\n    UCSR0B = (1 << TXEN0);\n    // 8-bit data, 1 stop bit\n    UCSR0C = (1 << UCSZ01) | (1 << UCSZ00);",
                "send_code": '    const char *msg = "Hello AVR!\\r\\n";\n    while (*msg) {\n        while (!(UCSR0A & (1 << UDRE0)));\n        UDR0 = *msg++;\n    }',
                "delay": "    _delay_ms(1000);",
            },
        ],
    },
    {
        "pattern": "i2c_read",
        "instruction": "Write C++ code to read data from a {sensor} via I2C on {platform} at address {addr}.",
        "code": """\
#include <{header}>
{extra_includes}

{defines}

void setup() {{
    {setup_code}
}}

void loop() {{
    {read_code}
    {delay}
}}""",
        "variants": [
            {
                "sensor": "temperature sensor (LM75)", "platform": "Arduino",
                "addr": "0x48",
                "header": "Arduino.h",
                "extra_includes": "#include <Wire.h>",
                "defines": "#define LM75_ADDR 0x48",
                "setup_code": "Serial.begin(9600);\n    Wire.begin();",
                "read_code": '    Wire.beginTransmission(LM75_ADDR);\n    Wire.write(0x00);  // Temperature register\n    Wire.endTransmission();\n    Wire.requestFrom(LM75_ADDR, 2);\n    if (Wire.available() >= 2) {\n        int16_t raw = (Wire.read() << 8) | Wire.read();\n        float temp = (raw >> 5) * 0.125;\n        Serial.print("Temp: ");\n        Serial.println(temp);\n    }',
                "delay": "    delay(1000);",
            },
            {
                "sensor": "accelerometer (MPU6050)", "platform": "ESP32",
                "addr": "0x68",
                "header": "Arduino.h",
                "extra_includes": "#include <Wire.h>",
                "defines": "#define MPU6050_ADDR 0x68",
                "setup_code": "Serial.begin(115200);\n    Wire.begin(21, 22);  // ESP32 I2C pins\n    // Wake up MPU6050\n    Wire.beginTransmission(MPU6050_ADDR);\n    Wire.write(0x6B);  // PWR_MGMT_1\n    Wire.write(0x00);\n    Wire.endTransmission();",
                "read_code": '    Wire.beginTransmission(MPU6050_ADDR);\n    Wire.write(0x3B);  // ACCEL_XOUT_H\n    Wire.endTransmission(false);\n    Wire.requestFrom((uint8_t)MPU6050_ADDR, (uint8_t)6);\n    int16_t ax = (Wire.read() << 8) | Wire.read();\n    int16_t ay = (Wire.read() << 8) | Wire.read();\n    int16_t az = (Wire.read() << 8) | Wire.read();\n    Serial.printf("Accel: X=%d Y=%d Z=%d\\n", ax, ay, az);',
                "delay": "    delay(100);",
            },
        ],
    },
    {
        "pattern": "spi_transfer",
        "instruction": "Write C++ code to communicate with a {device} via SPI on {platform}.",
        "code": """\
#include <{header}>
#include <SPI.h>

{defines}

void setup() {{
    {setup_code}
}}

{helper_funcs}

void loop() {{
    {loop_code}
    {delay}
}}""",
        "variants": [
            {
                "device": "MCP3008 ADC", "platform": "Arduino",
                "header": "Arduino.h",
                "defines": "const int CS_PIN = 10;",
                "setup_code": "Serial.begin(9600);\n    SPI.begin();\n    pinMode(CS_PIN, OUTPUT);\n    digitalWrite(CS_PIN, HIGH);",
                "helper_funcs": "uint16_t readADC(uint8_t channel) {\n    digitalWrite(CS_PIN, LOW);\n    SPI.transfer(0x01);  // Start bit\n    uint8_t hi = SPI.transfer((0x08 | channel) << 4);\n    uint8_t lo = SPI.transfer(0x00);\n    digitalWrite(CS_PIN, HIGH);\n    return ((hi & 0x03) << 8) | lo;\n}",
                "loop_code": '    uint16_t val = readADC(0);\n    Serial.print("ADC Ch0: ");\n    Serial.println(val);',
                "delay": "    delay(500);",
            },
            {
                "device": "MAX7219 LED matrix driver", "platform": "Arduino",
                "header": "Arduino.h",
                "defines": "const int CS_PIN = 10;\nconst int NUM_DEVICES = 1;",
                "setup_code": "SPI.begin();\n    pinMode(CS_PIN, OUTPUT);\n    // Initialize MAX7219\n    sendCommand(0x0C, 0x01);  // Shutdown register: normal operation\n    sendCommand(0x0B, 0x07);  // Scan limit: all 8 digits\n    sendCommand(0x09, 0x00);  // Decode mode: no decode\n    sendCommand(0x0A, 0x08);  // Intensity: medium\n    sendCommand(0x0F, 0x00);  // Display test: off",
                "helper_funcs": "void sendCommand(uint8_t reg, uint8_t data) {\n    digitalWrite(CS_PIN, LOW);\n    SPI.transfer(reg);\n    SPI.transfer(data);\n    digitalWrite(CS_PIN, HIGH);\n}",
                "loop_code": "    // Display a pattern on row 1\n    sendCommand(0x01, 0b10101010);",
                "delay": "    delay(1000);",
            },
        ],
    },
    {
        "pattern": "adc_read",
        "instruction": "Write C++ code to read an analog {sensor} using the ADC on {platform} and convert to {unit}.",
        "code": """\
#include <{header}>
{extra_includes}

{defines}

void setup() {{
    {setup_code}
}}

void loop() {{
    {read_code}
    {delay}
}}""",
        "variants": [
            {
                "sensor": "thermistor (NTC 10K)", "platform": "Arduino", "unit": "temperature in Celsius",
                "header": "Arduino.h", "extra_includes": "#include <math.h>",
                "defines": "const int THERM_PIN = A0;\nconst float R_FIXED = 10000.0;\nconst float BETA = 3950.0;\nconst float T0 = 298.15;\nconst float R0 = 10000.0;",
                "setup_code": "Serial.begin(9600);",
                "read_code": '    int raw = analogRead(THERM_PIN);\n    float resistance = R_FIXED * (1023.0 / raw - 1.0);\n    float tempK = 1.0 / (1.0/T0 + (1.0/BETA) * log(resistance/R0));\n    float tempC = tempK - 273.15;\n    Serial.print("Temperature: ");\n    Serial.print(tempC);\n    Serial.println(" C");',
                "delay": "    delay(1000);",
            },
            {
                "sensor": "potentiometer", "platform": "ESP32", "unit": "voltage",
                "header": "Arduino.h", "extra_includes": "",
                "defines": "const int POT_PIN = 34;\nconst float V_REF = 3.3;\nconst int ADC_MAX = 4095;",
                "setup_code": "Serial.begin(115200);\n    analogReadResolution(12);  // ESP32 12-bit ADC",
                "read_code": '    int raw = analogRead(POT_PIN);\n    float voltage = (raw / (float)ADC_MAX) * V_REF;\n    Serial.printf("ADC: %d  Voltage: %.2fV\\n", raw, voltage);',
                "delay": "    delay(200);",
            },
        ],
    },
    {
        "pattern": "pwm_output",
        "instruction": "Write C++ code to generate a PWM signal on {platform} to control a {actuator}.",
        "code": """\
#include <{header}>
{extra_includes}

{defines}

void setup() {{
    {setup_code}
}}

void loop() {{
    {loop_code}
}}""",
        "variants": [
            {
                "actuator": "servo motor", "platform": "Arduino",
                "header": "Arduino.h", "extra_includes": "#include <Servo.h>",
                "defines": "Servo myServo;\nconst int SERVO_PIN = 9;",
                "setup_code": "myServo.attach(SERVO_PIN);",
                "loop_code": "    // Sweep from 0 to 180 degrees\n    for (int angle = 0; angle <= 180; angle += 5) {\n        myServo.write(angle);\n        delay(30);\n    }\n    for (int angle = 180; angle >= 0; angle -= 5) {\n        myServo.write(angle);\n        delay(30);\n    }",
            },
            {
                "actuator": "DC motor (speed control via L298N)", "platform": "Arduino",
                "header": "Arduino.h", "extra_includes": "",
                "defines": "const int ENA = 9;   // PWM pin\nconst int IN1 = 8;\nconst int IN2 = 7;",
                "setup_code": "pinMode(ENA, OUTPUT);\n    pinMode(IN1, OUTPUT);\n    pinMode(IN2, OUTPUT);\n    // Set direction: forward\n    digitalWrite(IN1, HIGH);\n    digitalWrite(IN2, LOW);",
                "loop_code": "    // Ramp speed up\n    for (int speed = 0; speed <= 255; speed += 5) {\n        analogWrite(ENA, speed);\n        delay(50);\n    }\n    delay(2000);\n    // Ramp speed down\n    for (int speed = 255; speed >= 0; speed -= 5) {\n        analogWrite(ENA, speed);\n        delay(50);\n    }\n    delay(1000);",
            },
            {
                "actuator": "LED brightness (fade)", "platform": "ESP32",
                "header": "Arduino.h", "extra_includes": "",
                "defines": "const int LED_PIN = 16;\nconst int PWM_CHANNEL = 0;\nconst int PWM_FREQ = 5000;\nconst int PWM_RESOLUTION = 8;  // 8-bit: 0-255",
                "setup_code": "ledcSetup(PWM_CHANNEL, PWM_FREQ, PWM_RESOLUTION);\n    ledcAttachPin(LED_PIN, PWM_CHANNEL);",
                "loop_code": "    // Fade in\n    for (int duty = 0; duty <= 255; duty++) {\n        ledcWrite(PWM_CHANNEL, duty);\n        delay(10);\n    }\n    // Fade out\n    for (int duty = 255; duty >= 0; duty--) {\n        ledcWrite(PWM_CHANNEL, duty);\n        delay(10);\n    }",
            },
        ],
    },
    {
        "pattern": "timer_interrupt",
        "instruction": "Write C++ code to set up a timer interrupt on {platform} that fires every {interval}.",
        "code": """\
#include <{header}>
{extra_includes}

{defines}

{isr_code}

void setup() {{
    {setup_code}
}}

void loop() {{
    {loop_code}
}}""",
        "variants": [
            {
                "interval": "1 second", "platform": "ATmega328P (Timer1 CTC mode)",
                "header": "avr/io.h>\n#include <avr/interrupt.h",
                "extra_includes": "",
                "defines": "volatile uint16_t tick_count = 0;",
                "isr_code": "ISR(TIMER1_COMPA_vect) {\n    tick_count++;\n    PORTB ^= (1 << PB5);  // Toggle LED on PB5\n}",
                "setup_code": "DDRB |= (1 << PB5);  // LED output\n    // Timer1 CTC mode, prescaler 256\n    TCCR1A = 0;\n    TCCR1B = (1 << WGM12) | (1 << CS12);  // CTC, /256\n    OCR1A = 62499;  // 16MHz / 256 / 62500 = 1Hz\n    TIMSK1 = (1 << OCIE1A);  // Enable compare interrupt\n    sei();  // Enable global interrupts",
                "loop_code": "    // Main loop: CPU free for other tasks\n    // LED toggling handled in ISR",
            },
            {
                "interval": "100 milliseconds", "platform": "Arduino (Timer2)",
                "header": "Arduino.h",
                "extra_includes": "",
                "defines": "volatile uint32_t isr_counter = 0;",
                "isr_code": "ISR(TIMER2_COMPA_vect) {\n    isr_counter++;\n}",
                "setup_code": "Serial.begin(9600);\n    // Timer2 CTC mode\n    TCCR2A = (1 << WGM21);  // CTC mode\n    TCCR2B = (1 << CS22) | (1 << CS21) | (1 << CS20);  // /1024\n    OCR2A = 155;  // 16MHz / 1024 / 156 ≈ 100Hz\n    TIMSK2 = (1 << OCIE2A);\n    sei();",
                "loop_code": '    static uint32_t last = 0;\n    uint32_t current;\n    noInterrupts();\n    current = isr_counter;\n    interrupts();\n    if (current != last) {\n        Serial.print("ISR count: ");\n        Serial.println(current);\n        last = current;\n    }',
            },
            {
                "interval": "500 microseconds", "platform": "ESP32 (hw_timer)",
                "header": "Arduino.h",
                "extra_includes": "",
                "defines": "hw_timer_t *timer = NULL;\nvolatile bool flag = false;",
                "isr_code": "void IRAM_ATTR onTimer() {\n    flag = true;\n}",
                "setup_code": "Serial.begin(115200);\n    timer = timerBegin(0, 80, true);  // 80 prescaler → 1MHz\n    timerAttachInterrupt(timer, &onTimer, true);\n    timerAlarmWrite(timer, 500, true);  // 500us\n    timerAlarmEnable(timer);",
                "loop_code": '    if (flag) {\n        flag = false;\n        // Handle timer event\n        Serial.println("Timer fired!");\n    }',
            },
        ],
    },
    {
        "pattern": "button_debounce",
        "instruction": "Write C++ code to read a {component} with debouncing on {platform} using {method}.",
        "code": """\
#include <{header}>

{defines}

void setup() {{
    {setup_code}
}}

void loop() {{
    {loop_code}
}}""",
        "variants": [
            {
                "component": "push button", "platform": "Arduino", "method": "millis-based debounce",
                "header": "Arduino.h",
                "defines": "const int BUTTON_PIN = 2;\nconst int LED_PIN = 13;\nconst unsigned long DEBOUNCE_MS = 50;\n\nbool ledState = false;\nbool lastButtonState = HIGH;\nunsigned long lastDebounceTime = 0;",
                "setup_code": "pinMode(BUTTON_PIN, INPUT_PULLUP);\n    pinMode(LED_PIN, OUTPUT);",
                "loop_code": "    bool reading = digitalRead(BUTTON_PIN);\n    if (reading != lastButtonState) {\n        lastDebounceTime = millis();\n    }\n    if ((millis() - lastDebounceTime) > DEBOUNCE_MS) {\n        static bool buttonState = HIGH;\n        if (reading != buttonState) {\n            buttonState = reading;\n            if (buttonState == LOW) {\n                ledState = !ledState;\n                digitalWrite(LED_PIN, ledState);\n            }\n        }\n    }\n    lastButtonState = reading;",
            },
            {
                "component": "rotary encoder", "platform": "Arduino", "method": "interrupt-driven",
                "header": "Arduino.h",
                "defines": "const int CLK_PIN = 2;\nconst int DT_PIN = 3;\n\nvolatile int encoderCount = 0;",
                "setup_code": "Serial.begin(9600);\n    pinMode(CLK_PIN, INPUT_PULLUP);\n    pinMode(DT_PIN, INPUT_PULLUP);\n    attachInterrupt(digitalPinToInterrupt(CLK_PIN), readEncoder, FALLING);",
                "loop_code": '    static int lastCount = 0;\n    noInterrupts();\n    int count = encoderCount;\n    interrupts();\n    if (count != lastCount) {\n        Serial.print("Encoder: ");\n        Serial.println(count);\n        lastCount = count;\n    }',
            },
        ],
    },
    {
        "pattern": "display_output",
        "instruction": "Write C++ code to display {content} on a {display} connected via {interface} on {platform}.",
        "code": """\
#include <{header}>
{extra_includes}

{defines}

void setup() {{
    {setup_code}
}}

void loop() {{
    {loop_code}
    {delay}
}}""",
        "variants": [
            {
                "content": "text and sensor values", "display": "16x2 LCD", "interface": "I2C", "platform": "Arduino",
                "header": "Arduino.h",
                "extra_includes": "#include <Wire.h>\n#include <LiquidCrystal_I2C.h>",
                "defines": "LiquidCrystal_I2C lcd(0x27, 16, 2);\nconst int SENSOR_PIN = A0;",
                "setup_code": "lcd.init();\n    lcd.backlight();\n    lcd.setCursor(0, 0);\n    lcd.print(\"Sensor Monitor\");",
                "loop_code": '    int val = analogRead(SENSOR_PIN);\n    float voltage = val * (5.0 / 1023.0);\n    lcd.setCursor(0, 1);\n    lcd.print("V: ");\n    lcd.print(voltage, 2);\n    lcd.print("V   ");',
                "delay": "    delay(250);",
            },
            {
                "content": "graphics and text", "display": "0.96 inch OLED (SSD1306)", "interface": "I2C", "platform": "ESP32",
                "header": "Arduino.h",
                "extra_includes": "#include <Wire.h>\n#include <Adafruit_GFX.h>\n#include <Adafruit_SSD1306.h>",
                "defines": "#define SCREEN_WIDTH 128\n#define SCREEN_HEIGHT 64\n#define OLED_ADDR 0x3C\nAdafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire);",
                "setup_code": "Serial.begin(115200);\n    Wire.begin(21, 22);\n    if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {\n        Serial.println(\"SSD1306 init failed!\");\n        while (true);\n    }\n    display.clearDisplay();\n    display.setTextSize(1);\n    display.setTextColor(SSD1306_WHITE);",
                "loop_code": '    display.clearDisplay();\n    display.setCursor(0, 0);\n    display.println("ESP32 OLED Demo");\n    display.setCursor(0, 20);\n    display.print("Uptime: ");\n    display.print(millis() / 1000);\n    display.println("s");\n    display.display();',
                "delay": "    delay(500);",
            },
        ],
    },
    {
        "pattern": "freertos_task",
        "instruction": "Write C++ code to create {task_desc} using FreeRTOS on {platform}.",
        "code": """\
#include <{header}>
{extra_includes}

{defines}

{task_functions}

void setup() {{
    {setup_code}
}}

void loop() {{
    {loop_code}
}}""",
        "variants": [
            {
                "task_desc": "two tasks: one blinking an LED, another reading a sensor", "platform": "ESP32",
                "header": "Arduino.h",
                "extra_includes": "",
                "defines": "const int LED_PIN = 2;\nconst int SENSOR_PIN = 34;\n\nTaskHandle_t blinkTask;\nTaskHandle_t sensorTask;",
                "task_functions": 'void blinkTaskFunc(void *param) {\n    pinMode(LED_PIN, OUTPUT);\n    for (;;) {\n        digitalWrite(LED_PIN, HIGH);\n        vTaskDelay(pdMS_TO_TICKS(500));\n        digitalWrite(LED_PIN, LOW);\n        vTaskDelay(pdMS_TO_TICKS(500));\n    }\n}\n\nvoid sensorTaskFunc(void *param) {\n    for (;;) {\n        int val = analogRead(SENSOR_PIN);\n        Serial.printf("Sensor: %d\\n", val);\n        vTaskDelay(pdMS_TO_TICKS(1000));\n    }\n}',
                "setup_code": 'Serial.begin(115200);\n    xTaskCreatePinnedToCore(blinkTaskFunc, "Blink", 2048, NULL, 1, &blinkTask, 0);\n    xTaskCreatePinnedToCore(sensorTaskFunc, "Sensor", 4096, NULL, 1, &sensorTask, 1);',
                "loop_code": "    // FreeRTOS handles scheduling\n    vTaskDelay(pdMS_TO_TICKS(10000));",
            },
            {
                "task_desc": "a producer-consumer pattern with a queue", "platform": "ESP32",
                "header": "Arduino.h",
                "extra_includes": "",
                "defines": "QueueHandle_t dataQueue;\nconst int QUEUE_SIZE = 10;",
                "task_functions": 'void producerTask(void *param) {\n    int count = 0;\n    for (;;) {\n        int data = analogRead(34);\n        if (xQueueSend(dataQueue, &data, pdMS_TO_TICKS(100)) == pdTRUE) {\n            count++;\n        }\n        vTaskDelay(pdMS_TO_TICKS(200));\n    }\n}\n\nvoid consumerTask(void *param) {\n    int received;\n    for (;;) {\n        if (xQueueReceive(dataQueue, &received, pdMS_TO_TICKS(500)) == pdTRUE) {\n            Serial.printf("Consumed: %d\\n", received);\n        }\n    }\n}',
                "setup_code": 'Serial.begin(115200);\n    dataQueue = xQueueCreate(QUEUE_SIZE, sizeof(int));\n    xTaskCreate(producerTask, "Producer", 4096, NULL, 1, NULL);\n    xTaskCreate(consumerTask, "Consumer", 4096, NULL, 1, NULL);',
                "loop_code": "    vTaskDelay(pdMS_TO_TICKS(60000));",
            },
        ],
    },
    {
        "pattern": "relay_control",
        "instruction": "Write C++ code to control a {load} via relay on {platform} based on {trigger}.",
        "code": """\
#include <{header}>

{defines}

void setup() {{
    {setup_code}
}}

void loop() {{
    {loop_code}
    {delay}
}}""",
        "variants": [
            {
                "load": "water pump", "platform": "Arduino", "trigger": "soil moisture sensor threshold",
                "header": "Arduino.h",
                "defines": "const int RELAY_PIN = 7;\nconst int MOISTURE_PIN = A0;\nconst int DRY_THRESHOLD = 400;  // Below this = dry",
                "setup_code": "Serial.begin(9600);\n    pinMode(RELAY_PIN, OUTPUT);\n    digitalWrite(RELAY_PIN, HIGH);  // Relay off (active-low)",
                "loop_code": '    int moisture = analogRead(MOISTURE_PIN);\n    Serial.print("Moisture: ");\n    Serial.println(moisture);\n    if (moisture < DRY_THRESHOLD) {\n        digitalWrite(RELAY_PIN, LOW);  // Pump ON\n        Serial.println("Pump ON - watering");\n    } else {\n        digitalWrite(RELAY_PIN, HIGH);  // Pump OFF\n        Serial.println("Pump OFF");\n    }',
                "delay": "    delay(2000);",
            },
            {
                "load": "heating element", "platform": "ESP32", "trigger": "temperature reading from DHT22",
                "header": "Arduino.h",
                "defines": '#include "DHT.h"\n\nconst int RELAY_PIN = 16;\nconst int DHT_PIN = 4;\nconst float TARGET_TEMP = 22.0;\nconst float HYSTERESIS = 1.0;\n\nDHT dht(DHT_PIN, DHT22);',
                "setup_code": "Serial.begin(115200);\n    pinMode(RELAY_PIN, OUTPUT);\n    dht.begin();",
                "loop_code": '    float temp = dht.readTemperature();\n    if (isnan(temp)) {\n        Serial.println("DHT read error!");\n        return;\n    }\n    Serial.printf("Temp: %.1f C\\n", temp);\n    if (temp < TARGET_TEMP - HYSTERESIS) {\n        digitalWrite(RELAY_PIN, LOW);   // Heater ON\n    } else if (temp > TARGET_TEMP + HYSTERESIS) {\n        digitalWrite(RELAY_PIN, HIGH);  // Heater OFF\n    }',
                "delay": "    delay(5000);",
            },
        ],
    },
]


def generate_combinatorial_synthetics(rng: random.Random, target: int = 1200) -> list[dict]:
    """Generate combinatorial embedded C++ instruction pairs.

    Uses parameterized templates with multiple axes of variation
    (platform x peripheral x sensor x protocol) to create diverse
    but realistic embedded code examples.
    """
    records: list[dict] = []

    # ── Axis definitions ──────────────────────────────────────────────
    platforms = [
        {"name": "Arduino Uno (ATmega328P)", "header": "Arduino.h", "serial_init": "Serial.begin(9600);", "adc_bits": 10, "adc_max": 1023, "vref": 5.0, "delay_fn": "delay", "i2c_init": "Wire.begin();", "spi_init": "SPI.begin();"},
        {"name": "Arduino Mega (ATmega2560)", "header": "Arduino.h", "serial_init": "Serial.begin(9600);", "adc_bits": 10, "adc_max": 1023, "vref": 5.0, "delay_fn": "delay", "i2c_init": "Wire.begin();", "spi_init": "SPI.begin();"},
        {"name": "ESP32", "header": "Arduino.h", "serial_init": "Serial.begin(115200);", "adc_bits": 12, "adc_max": 4095, "vref": 3.3, "delay_fn": "delay", "i2c_init": "Wire.begin(21, 22);", "spi_init": "SPI.begin();"},
        {"name": "ESP8266 (NodeMCU)", "header": "Arduino.h", "serial_init": "Serial.begin(115200);", "adc_bits": 10, "adc_max": 1023, "vref": 3.3, "delay_fn": "delay", "i2c_init": "Wire.begin(D2, D1);", "spi_init": "SPI.begin();"},
        {"name": "STM32 (Blue Pill)", "header": "Arduino.h", "serial_init": "Serial.begin(115200);", "adc_bits": 12, "adc_max": 4095, "vref": 3.3, "delay_fn": "delay", "i2c_init": "Wire.begin();", "spi_init": "SPI.begin();"},
    ]

    sensors = [
        {"name": "DHT22 temperature/humidity sensor", "lib": "DHT.h", "read_code": 'DHT dht(SENSOR_PIN, DHT22);\n\n    dht.begin();\n    float temp = dht.readTemperature();\n    float hum = dht.readHumidity();', "print": 'Serial.printf("Temp: %.1fC  Hum: %.1f%%\\n", temp, hum);'},
        {"name": "BMP280 barometric pressure sensor", "lib": "Adafruit_BMP280.h", "read_code": 'Adafruit_BMP280 bmp;\n    bmp.begin(0x76);', "print": 'Serial.printf("Temp: %.1fC  Pressure: %.0f Pa\\n", bmp.readTemperature(), bmp.readPressure());'},
        {"name": "MPU6050 accelerometer/gyroscope", "lib": "Wire.h", "read_code": "Wire.beginTransmission(0x68);\n    Wire.write(0x6B);\n    Wire.write(0);\n    Wire.endTransmission();", "print": 'Serial.printf("Accel X=%d Y=%d Z=%d\\n", ax, ay, az);'},
        {"name": "HC-SR04 ultrasonic distance sensor", "lib": "", "read_code": "pinMode(TRIG_PIN, OUTPUT);\n    pinMode(ECHO_PIN, INPUT);", "print": 'Serial.printf("Distance: %.1f cm\\n", distance);'},
        {"name": "DS18B20 one-wire temperature sensor", "lib": "OneWire.h", "read_code": "OneWire oneWire(SENSOR_PIN);\n    DallasTemperature sensors(&oneWire);\n    sensors.begin();", "print": 'Serial.printf("Temp: %.2fC\\n", sensors.getTempCByIndex(0));'},
        {"name": "LDR (light dependent resistor)", "lib": "", "read_code": "// LDR connected to ADC pin with voltage divider", "print": 'Serial.printf("Light level: %d\\n", analogRead(SENSOR_PIN));'},
        {"name": "MQ-2 gas/smoke sensor", "lib": "", "read_code": "// MQ-2 analog output connected to ADC", "print": 'Serial.printf("Gas level: %d\\n", analogRead(GAS_PIN));'},
        {"name": "IR proximity sensor (TCRT5000)", "lib": "", "read_code": "pinMode(IR_PIN, INPUT);", "print": 'Serial.printf("Object detected: %s\\n", digitalRead(IR_PIN) ? "No" : "Yes");'},
    ]

    actuators = [
        {"name": "servo motor (SG90)", "lib": "Servo.h", "setup": "Servo myServo;\n    myServo.attach(SERVO_PIN);", "action": "myServo.write(angle);"},
        {"name": "DC motor via L298N H-bridge", "lib": "", "setup": "pinMode(ENA, OUTPUT);\n    pinMode(IN1, OUTPUT);\n    pinMode(IN2, OUTPUT);", "action": "analogWrite(ENA, speed);\n    digitalWrite(IN1, HIGH);\n    digitalWrite(IN2, LOW);"},
        {"name": "stepper motor (28BYJ-48 + ULN2003)", "lib": "Stepper.h", "setup": "Stepper myStepper(2048, 8, 10, 9, 11);\n    myStepper.setSpeed(15);", "action": "myStepper.step(stepsToMove);"},
        {"name": "buzzer/piezo speaker", "lib": "", "setup": "pinMode(BUZZER_PIN, OUTPUT);", "action": "tone(BUZZER_PIN, frequency, duration);"},
        {"name": "relay module", "lib": "", "setup": "pinMode(RELAY_PIN, OUTPUT);\n    digitalWrite(RELAY_PIN, HIGH);  // OFF (active-low)", "action": "digitalWrite(RELAY_PIN, state);"},
        {"name": "RGB LED (common cathode)", "lib": "", "setup": "pinMode(RED_PIN, OUTPUT);\n    pinMode(GREEN_PIN, OUTPUT);\n    pinMode(BLUE_PIN, OUTPUT);", "action": "analogWrite(RED_PIN, r);\n    analogWrite(GREEN_PIN, g);\n    analogWrite(BLUE_PIN, b);"},
        {"name": "NeoPixel LED strip (WS2812B)", "lib": "Adafruit_NeoPixel.h", "setup": "Adafruit_NeoPixel strip(NUM_LEDS, LED_PIN, NEO_GRB + NEO_KHZ800);\n    strip.begin();\n    strip.show();", "action": "strip.setPixelColor(i, strip.Color(r, g, b));\n    strip.show();"},
    ]

    protocols = [
        {"name": "I2C", "includes": "#include <Wire.h>", "init": "Wire.begin();"},
        {"name": "SPI", "includes": "#include <SPI.h>", "init": "SPI.begin();"},
        {"name": "UART/Serial", "includes": "", "init": "Serial.begin(115200);"},
        {"name": "One-Wire", "includes": "#include <OneWire.h>", "init": ""},
    ]

    # ── Pattern 1: Read sensor, print value ───────────────────────────
    for plat in platforms:
        for sens in sensors:
            instruction = f"Write embedded C++ code to read a {sens['name']} on {plat['name']} and print the value over Serial."
            lib_include = f'\n#include "{sens["lib"]}"' if sens["lib"] else ""
            code = f"""#include <{plat['header']}>{lib_include}

const int SENSOR_PIN = A0;

void setup() {{
    {plat['serial_init']}
    {sens['read_code']}
    Serial.println("{sens['name']} initialized");
}}

void loop() {{
    {sens['print']}
    {plat['delay_fn']}(1000);
}}"""
            records.append(make_msg(instruction, code, {
                "source": "synthetic-combinatorial",
                "license": "CC0-1.0",
                "pattern": "sensor-read",
                "platform": plat["name"],
                "domain_tag": "synthetic-embedded",
            }))

    # ── Pattern 2: Sensor → actuator control loop ─────────────────────
    combos = [(s, a) for s in sensors[:5] for a in actuators[:5]]
    rng.shuffle(combos)
    for sens, act in combos[:20]:
        for plat in rng.sample(platforms, 2):
            instruction = f"Write C++ firmware for {plat['name']} that reads a {sens['name']} and controls a {act['name']} based on the reading."
            lib_includes = ""
            if sens["lib"]:
                lib_includes += f'\n#include "{sens["lib"]}"'
            if act["lib"]:
                lib_includes += f'\n#include "{act["lib"]}"'
            code = f"""#include <{plat['header']}>{lib_includes}

const int SENSOR_PIN = A0;
const int ACTUATOR_PIN = 9;
const int THRESHOLD = {plat['adc_max'] // 2};

void setup() {{
    {plat['serial_init']}
    {sens['read_code']}
    {act['setup']}
}}

void loop() {{
    int sensorVal = analogRead(SENSOR_PIN);
    {sens['print']}

    if (sensorVal > THRESHOLD) {{
        {act['action']}
        Serial.println("Actuator activated");
    }}
    {plat['delay_fn']}(500);
}}"""
            records.append(make_msg(instruction, code, {
                "source": "synthetic-combinatorial",
                "license": "CC0-1.0",
                "pattern": "sensor-actuator",
                "platform": plat["name"],
                "domain_tag": "synthetic-embedded",
            }))

    # ── Pattern 3: Protocol-specific init + read ──────────────────────
    i2c_devices = [
        ("BMP280 (0x76)", "0x76", "temperature and pressure"),
        ("SHT31 (0x44)", "0x44", "temperature and humidity"),
        ("ADS1115 (0x48)", "0x48", "16-bit ADC values"),
        ("PCF8574 (0x20)", "0x20", "I/O expander state"),
        ("MCP23017 (0x20)", "0x20", "16-bit I/O expander"),
        ("VEML7700 (0x10)", "0x10", "ambient light level"),
        ("BME680 (0x77)", "0x77", "environmental data"),
        ("INA219 (0x40)", "0x40", "current and voltage"),
    ]
    for plat in platforms:
        for dev_name, addr, reading in i2c_devices:
            instruction = f"Write C++ code to read {reading} from a {dev_name} sensor via I2C on {plat['name']}."
            code = f"""#include <{plat['header']}>
#include <Wire.h>

#define DEVICE_ADDR {addr}

void setup() {{
    {plat['serial_init']}
    {plat['i2c_init']}
    Serial.println("I2C {dev_name} ready");
}}

void loop() {{
    Wire.beginTransmission(DEVICE_ADDR);
    Wire.write(0x00);  // Data register
    Wire.endTransmission();

    Wire.requestFrom((uint8_t)DEVICE_ADDR, (uint8_t)2);
    if (Wire.available() >= 2) {{
        int16_t raw = (Wire.read() << 8) | Wire.read();
        Serial.print("{reading}: ");
        Serial.println(raw);
    }}
    {plat['delay_fn']}(1000);
}}"""
            records.append(make_msg(instruction, code, {
                "source": "synthetic-combinatorial",
                "license": "CC0-1.0",
                "pattern": "i2c-device",
                "platform": plat["name"],
                "domain_tag": "synthetic-embedded",
            }))

    # ── Pattern 4: Bare-metal AVR register manipulation ───────────────
    avr_patterns = [
        ("Set up Timer0 in CTC mode for a periodic interrupt on ATmega328P.",
         '#include <avr/io.h>\n#include <avr/interrupt.h>\n\nvolatile uint32_t ticks = 0;\n\nISR(TIMER0_COMPA_vect) {\n    ticks++;\n}\n\nint main(void) {\n    // Timer0 CTC mode, prescaler /64\n    TCCR0A = (1 << WGM01);\n    TCCR0B = (1 << CS01) | (1 << CS00);\n    OCR0A = 249;  // 16MHz/64/250 = 1kHz\n    TIMSK0 = (1 << OCIE0A);\n    sei();\n\n    DDRB |= (1 << PB5);  // LED output\n    while (1) {\n        if (ticks >= 1000) {\n            PORTB ^= (1 << PB5);\n            ticks = 0;\n        }\n    }\n}'),
        ("Configure ADC on ATmega328P to read channel 0 with 10-bit resolution.",
         '#include <avr/io.h>\n#include <util/delay.h>\n\nvoid adc_init(void) {\n    ADMUX = (1 << REFS0);  // AVCC reference\n    ADCSRA = (1 << ADEN) | (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0);  // /128 prescaler\n}\n\nuint16_t adc_read(uint8_t channel) {\n    ADMUX = (ADMUX & 0xF0) | (channel & 0x0F);\n    ADCSRA |= (1 << ADSC);  // Start conversion\n    while (ADCSRA & (1 << ADSC));  // Wait\n    return ADC;\n}\n\nint main(void) {\n    adc_init();\n    DDRB |= (1 << PB5);\n    while (1) {\n        uint16_t val = adc_read(0);\n        if (val > 512) {\n            PORTB |= (1 << PB5);\n        } else {\n            PORTB &= ~(1 << PB5);\n        }\n        _delay_ms(100);\n    }\n}'),
        ("Implement UART transmit and receive on ATmega328P at 9600 baud.",
         '#include <avr/io.h>\n#include <avr/interrupt.h>\n\n#define BAUD 9600\n#define UBRR_VAL ((F_CPU / (16UL * BAUD)) - 1)\n\nvoid uart_init(void) {\n    UBRR0H = (uint8_t)(UBRR_VAL >> 8);\n    UBRR0L = (uint8_t)UBRR_VAL;\n    UCSR0B = (1 << TXEN0) | (1 << RXEN0);\n    UCSR0C = (1 << UCSZ01) | (1 << UCSZ00);  // 8N1\n}\n\nvoid uart_putchar(char c) {\n    while (!(UCSR0A & (1 << UDRE0)));\n    UDR0 = c;\n}\n\nchar uart_getchar(void) {\n    while (!(UCSR0A & (1 << RXC0)));\n    return UDR0;\n}\n\nvoid uart_puts(const char *s) {\n    while (*s) uart_putchar(*s++);\n}\n\nint main(void) {\n    uart_init();\n    uart_puts("ATmega328P UART ready\\r\\n");\n    while (1) {\n        char c = uart_getchar();\n        uart_putchar(c);  // Echo\n    }\n}'),
        ("Configure external interrupt INT0 on ATmega328P for a button press (falling edge).",
         '#include <avr/io.h>\n#include <avr/interrupt.h>\n#include <util/delay.h>\n\nvolatile uint8_t button_pressed = 0;\n\nISR(INT0_vect) {\n    button_pressed = 1;\n}\n\nint main(void) {\n    DDRB |= (1 << PB5);   // LED output\n    DDRD &= ~(1 << PD2);  // INT0 input\n    PORTD |= (1 << PD2);  // Pull-up\n\n    // Falling edge trigger\n    EICRA = (1 << ISC01);\n    EIMSK = (1 << INT0);\n    sei();\n\n    while (1) {\n        if (button_pressed) {\n            PORTB ^= (1 << PB5);  // Toggle LED\n            button_pressed = 0;\n            _delay_ms(200);  // Debounce\n        }\n    }\n}'),
        ("Implement SPI master mode on ATmega328P to communicate with a peripheral.",
         '#include <avr/io.h>\n#include <util/delay.h>\n\n#define CS_PIN PB2\n#define MOSI_PIN PB3\n#define SCK_PIN PB5\n\nvoid spi_init(void) {\n    DDRB |= (1 << CS_PIN) | (1 << MOSI_PIN) | (1 << SCK_PIN);\n    PORTB |= (1 << CS_PIN);  // CS high (deselect)\n    SPCR = (1 << SPE) | (1 << MSTR) | (1 << SPR0);  // Enable, Master, /16\n}\n\nuint8_t spi_transfer(uint8_t data) {\n    SPDR = data;\n    while (!(SPSR & (1 << SPIF)));\n    return SPDR;\n}\n\nvoid spi_select(void) { PORTB &= ~(1 << CS_PIN); }\nvoid spi_deselect(void) { PORTB |= (1 << CS_PIN); }\n\nint main(void) {\n    spi_init();\n    while (1) {\n        spi_select();\n        uint8_t result = spi_transfer(0xAA);\n        spi_deselect();\n        _delay_ms(100);\n    }\n}'),
        ("Implement TWI (I2C) master on ATmega328P to read a byte from a slave device.",
         '#include <avr/io.h>\n#include <util/delay.h>\n\n#define TWI_FREQ 100000UL\n\nvoid twi_init(void) {\n    TWSR = 0;  // Prescaler = 1\n    TWBR = ((F_CPU / TWI_FREQ) - 16) / 2;\n}\n\nvoid twi_start(void) {\n    TWCR = (1 << TWINT) | (1 << TWSTA) | (1 << TWEN);\n    while (!(TWCR & (1 << TWINT)));\n}\n\nvoid twi_stop(void) {\n    TWCR = (1 << TWINT) | (1 << TWSTO) | (1 << TWEN);\n}\n\nvoid twi_write(uint8_t data) {\n    TWDR = data;\n    TWCR = (1 << TWINT) | (1 << TWEN);\n    while (!(TWCR & (1 << TWINT)));\n}\n\nuint8_t twi_read_nack(void) {\n    TWCR = (1 << TWINT) | (1 << TWEN);\n    while (!(TWCR & (1 << TWINT)));\n    return TWDR;\n}\n\nint main(void) {\n    twi_init();\n    while (1) {\n        twi_start();\n        twi_write(0xA0 | 1);  // Address + Read\n        uint8_t data = twi_read_nack();\n        twi_stop();\n        _delay_ms(500);\n    }\n}'),
        ("Set up PWM output on ATmega328P Timer1 for variable duty cycle.",
         '#include <avr/io.h>\n#include <util/delay.h>\n\nvoid pwm_init(void) {\n    // Fast PWM, 10-bit, non-inverting on OC1A (PB1)\n    DDRB |= (1 << PB1);\n    TCCR1A = (1 << COM1A1) | (1 << WGM11) | (1 << WGM10);\n    TCCR1B = (1 << WGM12) | (1 << CS11);  // /8 prescaler\n}\n\nvoid pwm_set_duty(uint16_t duty) {\n    OCR1A = duty;  // 0-1023\n}\n\nint main(void) {\n    pwm_init();\n    uint16_t duty = 0;\n    int8_t direction = 1;\n    while (1) {\n        pwm_set_duty(duty);\n        duty += direction * 10;\n        if (duty >= 1023) direction = -1;\n        if (duty == 0) direction = 1;\n        _delay_ms(20);\n    }\n}'),
        ("Configure watchdog timer on ATmega328P to reset the MCU after 2 seconds.",
         '#include <avr/io.h>\n#include <avr/wdt.h>\n#include <avr/interrupt.h>\n#include <util/delay.h>\n\nint main(void) {\n    // Disable watchdog on boot (safety)\n    MCUSR &= ~(1 << WDRF);\n    wdt_disable();\n\n    DDRB |= (1 << PB5);  // LED\n\n    // Enable watchdog: 2s timeout, reset mode\n    cli();\n    wdt_reset();\n    WDTCSR |= (1 << WDCE) | (1 << WDE);\n    WDTCSR = (1 << WDE) | (1 << WDP2) | (1 << WDP1) | (1 << WDP0);  // ~2s\n    sei();\n\n    // Main loop — must call wdt_reset() periodically\n    while (1) {\n        PORTB ^= (1 << PB5);\n        wdt_reset();  // Feed the watchdog\n        _delay_ms(500);\n    }\n}'),
    ]
    for instr, code in avr_patterns:
        records.append(make_msg(instr, code, {
            "source": "synthetic-avr-baremetal",
            "license": "CC0-1.0",
            "pattern": "avr-register",
            "domain_tag": "synthetic-embedded",
        }))

    # ── Pattern 5: STM32 HAL patterns ─────────────────────────────────
    stm32_patterns = [
        ("Write STM32 HAL code to toggle an LED on PA5 using GPIO.",
         '#include "stm32f4xx_hal.h"\n\nvoid SystemClock_Config(void);\nstatic void MX_GPIO_Init(void);\n\nint main(void) {\n    HAL_Init();\n    SystemClock_Config();\n    MX_GPIO_Init();\n\n    while (1) {\n        HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);\n        HAL_Delay(500);\n    }\n}\n\nstatic void MX_GPIO_Init(void) {\n    __HAL_RCC_GPIOA_CLK_ENABLE();\n    GPIO_InitTypeDef GPIO_InitStruct = {0};\n    GPIO_InitStruct.Pin = GPIO_PIN_5;\n    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;\n    GPIO_InitStruct.Pull = GPIO_NOPULL;\n    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;\n    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);\n}'),
        ("Write STM32 HAL code to read ADC channel 0 and print via UART.",
         '#include "stm32f4xx_hal.h"\n#include <stdio.h>\n\nADC_HandleTypeDef hadc1;\nUART_HandleTypeDef huart2;\n\nvoid SystemClock_Config(void);\nstatic void MX_ADC1_Init(void);\nstatic void MX_USART2_Init(void);\n\nint main(void) {\n    HAL_Init();\n    SystemClock_Config();\n    MX_ADC1_Init();\n    MX_USART2_Init();\n\n    char buf[64];\n    while (1) {\n        HAL_ADC_Start(&hadc1);\n        HAL_ADC_PollForConversion(&hadc1, 100);\n        uint32_t adc_val = HAL_ADC_GetValue(&hadc1);\n        HAL_ADC_Stop(&hadc1);\n\n        int len = snprintf(buf, sizeof(buf), "ADC: %lu\\r\\n", adc_val);\n        HAL_UART_Transmit(&huart2, (uint8_t*)buf, len, 100);\n        HAL_Delay(500);\n    }\n}'),
        ("Write STM32 HAL code to generate a PWM signal on TIM3 Channel 1.",
         '#include "stm32f4xx_hal.h"\n\nTIM_HandleTypeDef htim3;\n\nvoid SystemClock_Config(void);\nstatic void MX_TIM3_Init(void);\n\nint main(void) {\n    HAL_Init();\n    SystemClock_Config();\n    MX_TIM3_Init();\n\n    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1);\n\n    uint16_t duty = 0;\n    while (1) {\n        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, duty);\n        duty += 100;\n        if (duty > 9999) duty = 0;\n        HAL_Delay(50);\n    }\n}\n\nstatic void MX_TIM3_Init(void) {\n    __HAL_RCC_TIM3_CLK_ENABLE();\n    htim3.Instance = TIM3;\n    htim3.Init.Prescaler = 83;  // 84MHz / 84 = 1MHz\n    htim3.Init.CounterMode = TIM_COUNTERMODE_UP;\n    htim3.Init.Period = 9999;  // 1MHz / 10000 = 100Hz\n    HAL_TIM_PWM_Init(&htim3);\n\n    TIM_OC_InitTypeDef sConfig = {0};\n    sConfig.OCMode = TIM_OCMODE_PWM1;\n    sConfig.Pulse = 0;\n    HAL_TIM_PWM_ConfigChannel(&htim3, &sConfig, TIM_CHANNEL_1);\n}'),
        ("Write STM32 HAL code to configure EXTI interrupt on PA0 for a button press.",
         '#include "stm32f4xx_hal.h"\n\nvolatile uint8_t button_flag = 0;\n\nvoid SystemClock_Config(void);\nstatic void MX_GPIO_Init(void);\n\nvoid HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin) {\n    if (GPIO_Pin == GPIO_PIN_0) {\n        button_flag = 1;\n    }\n}\n\nint main(void) {\n    HAL_Init();\n    SystemClock_Config();\n    MX_GPIO_Init();\n\n    while (1) {\n        if (button_flag) {\n            HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);\n            button_flag = 0;\n            HAL_Delay(200);  // Debounce\n        }\n    }\n}\n\nstatic void MX_GPIO_Init(void) {\n    __HAL_RCC_GPIOA_CLK_ENABLE();\n\n    // LED on PA5\n    GPIO_InitTypeDef gpio = {0};\n    gpio.Pin = GPIO_PIN_5;\n    gpio.Mode = GPIO_MODE_OUTPUT_PP;\n    HAL_GPIO_Init(GPIOA, &gpio);\n\n    // Button on PA0 with EXTI\n    gpio.Pin = GPIO_PIN_0;\n    gpio.Mode = GPIO_MODE_IT_FALLING;\n    gpio.Pull = GPIO_PULLUP;\n    HAL_GPIO_Init(GPIOA, &gpio);\n\n    HAL_NVIC_SetPriority(EXTI0_IRQn, 2, 0);\n    HAL_NVIC_EnableIRQ(EXTI0_IRQn);\n}'),
        ("Write STM32 HAL code to transmit and receive data via SPI.",
         '#include "stm32f4xx_hal.h"\n\nSPI_HandleTypeDef hspi1;\n\nvoid SystemClock_Config(void);\nstatic void MX_SPI1_Init(void);\nstatic void MX_GPIO_Init(void);\n\n#define CS_PIN GPIO_PIN_4\n#define CS_PORT GPIOA\n\nvoid CS_Select(void) { HAL_GPIO_WritePin(CS_PORT, CS_PIN, GPIO_PIN_RESET); }\nvoid CS_Deselect(void) { HAL_GPIO_WritePin(CS_PORT, CS_PIN, GPIO_PIN_SET); }\n\nuint8_t SPI_ReadWrite(uint8_t data) {\n    uint8_t rx;\n    HAL_SPI_TransmitReceive(&hspi1, &data, &rx, 1, 100);\n    return rx;\n}\n\nint main(void) {\n    HAL_Init();\n    SystemClock_Config();\n    MX_GPIO_Init();\n    MX_SPI1_Init();\n\n    while (1) {\n        CS_Select();\n        uint8_t id = SPI_ReadWrite(0x9F);  // Read device ID\n        CS_Deselect();\n        HAL_Delay(500);\n    }\n}'),
        ("Write STM32 HAL code to use DMA for ADC continuous conversion.",
         '#include "stm32f4xx_hal.h"\n\nADC_HandleTypeDef hadc1;\nDMA_HandleTypeDef hdma_adc1;\n\n#define ADC_BUF_LEN 128\nuint32_t adc_buf[ADC_BUF_LEN];\nvolatile uint8_t adc_complete = 0;\n\nvoid HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc) {\n    adc_complete = 1;\n}\n\nvoid SystemClock_Config(void);\nstatic void MX_ADC1_Init(void);\nstatic void MX_DMA_Init(void);\n\nint main(void) {\n    HAL_Init();\n    SystemClock_Config();\n    MX_DMA_Init();\n    MX_ADC1_Init();\n\n    HAL_ADC_Start_DMA(&hadc1, adc_buf, ADC_BUF_LEN);\n\n    while (1) {\n        if (adc_complete) {\n            adc_complete = 0;\n            uint32_t sum = 0;\n            for (int i = 0; i < ADC_BUF_LEN; i++) sum += adc_buf[i];\n            uint32_t avg = sum / ADC_BUF_LEN;\n            // Process averaged ADC value\n        }\n    }\n}'),
    ]
    for instr, code in stm32_patterns:
        records.append(make_msg(instr, code, {
            "source": "synthetic-stm32-hal",
            "license": "CC0-1.0",
            "pattern": "stm32-hal",
            "domain_tag": "synthetic-embedded",
        }))

    # ── Pattern 6: ESP32 specific patterns ────────────────────────────
    esp32_patterns = [
        ("Write ESP32 code to create a WiFi access point and simple HTTP server.",
         '#include <Arduino.h>\n#include <WiFi.h>\n#include <WebServer.h>\n\nconst char *ssid = "ESP32_AP";\nconst char *password = "12345678";\nWebServer server(80);\n\nvoid handleRoot() {\n    int sensorVal = analogRead(34);\n    String html = "<html><body><h1>ESP32 Sensor</h1>";\n    html += "<p>ADC: " + String(sensorVal) + "</p>";\n    html += "</body></html>";\n    server.send(200, "text/html", html);\n}\n\nvoid setup() {\n    Serial.begin(115200);\n    WiFi.softAP(ssid, password);\n    Serial.print("AP IP: ");\n    Serial.println(WiFi.softAPIP());\n    server.on("/", handleRoot);\n    server.begin();\n}\n\nvoid loop() {\n    server.handleClient();\n}'),
        ("Write ESP32 code to use deep sleep with a timer wakeup and read a sensor on boot.",
         '#include <Arduino.h>\n\n#define uS_TO_S 1000000ULL\n#define SLEEP_SECONDS 30\n\nRTC_DATA_ATTR int bootCount = 0;\n\nvoid setup() {\n    Serial.begin(115200);\n    bootCount++;\n    Serial.printf("Boot #%d\\n", bootCount);\n\n    // Read sensor\n    int sensorVal = analogRead(34);\n    Serial.printf("Sensor: %d\\n", sensorVal);\n\n    // Configure wakeup\n    esp_sleep_enable_timer_wakeup(SLEEP_SECONDS * uS_TO_S);\n    Serial.printf("Sleeping for %d seconds...\\n", SLEEP_SECONDS);\n    Serial.flush();\n    esp_deep_sleep_start();\n}\n\nvoid loop() {\n    // Never reached\n}'),
        ("Write ESP32 code to send sensor data via BLE (Bluetooth Low Energy).",
         '#include <Arduino.h>\n#include <BLEDevice.h>\n#include <BLEServer.h>\n#include <BLEUtils.h>\n#include <BLE2902.h>\n\n#define SERVICE_UUID        "12345678-1234-1234-1234-123456789012"\n#define CHARACTERISTIC_UUID "12345678-1234-1234-1234-123456789013"\n\nBLECharacteristic *pCharacteristic;\nbool deviceConnected = false;\n\nclass MyServerCallbacks : public BLEServerCallbacks {\n    void onConnect(BLEServer *pServer) { deviceConnected = true; }\n    void onDisconnect(BLEServer *pServer) { deviceConnected = false; }\n};\n\nvoid setup() {\n    Serial.begin(115200);\n    BLEDevice::init("ESP32_Sensor");\n    BLEServer *pServer = BLEDevice::createServer();\n    pServer->setCallbacks(new MyServerCallbacks());\n    BLEService *pService = pServer->createService(SERVICE_UUID);\n    pCharacteristic = pService->createCharacteristic(\n        CHARACTERISTIC_UUID,\n        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY\n    );\n    pCharacteristic->addDescriptor(new BLE2902());\n    pService->start();\n    BLEAdvertising *pAdv = BLEDevice::getAdvertising();\n    pAdv->start();\n}\n\nvoid loop() {\n    if (deviceConnected) {\n        int val = analogRead(34);\n        char buf[8];\n        snprintf(buf, sizeof(buf), "%d", val);\n        pCharacteristic->setValue(buf);\n        pCharacteristic->notify();\n    }\n    delay(1000);\n}'),
        ("Write ESP32 code to use dual-core: task on Core 0 reads sensors, task on Core 1 handles display.",
         '#include <Arduino.h>\n#include <Wire.h>\n#include <Adafruit_SSD1306.h>\n\n#define SCREEN_WIDTH 128\n#define SCREEN_HEIGHT 64\nAdafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire);\n\nvolatile float temperature = 0;\nvolatile int lightLevel = 0;\nSemaphoreHandle_t dataMutex;\n\nvoid sensorTask(void *param) {\n    for (;;) {\n        float t = analogRead(34) * 0.1;  // Simplified\n        int l = analogRead(35);\n        xSemaphoreTake(dataMutex, portMAX_DELAY);\n        temperature = t;\n        lightLevel = l;\n        xSemaphoreGive(dataMutex);\n        vTaskDelay(pdMS_TO_TICKS(500));\n    }\n}\n\nvoid displayTask(void *param) {\n    display.begin(SSD1306_SWITCHCAPVCC, 0x3C);\n    for (;;) {\n        float t; int l;\n        xSemaphoreTake(dataMutex, portMAX_DELAY);\n        t = temperature;\n        l = lightLevel;\n        xSemaphoreGive(dataMutex);\n\n        display.clearDisplay();\n        display.setTextSize(1);\n        display.setTextColor(SSD1306_WHITE);\n        display.setCursor(0, 0);\n        display.printf("Temp: %.1f C\\nLight: %d", t, l);\n        display.display();\n        vTaskDelay(pdMS_TO_TICKS(250));\n    }\n}\n\nvoid setup() {\n    Serial.begin(115200);\n    Wire.begin(21, 22);\n    dataMutex = xSemaphoreCreateMutex();\n    xTaskCreatePinnedToCore(sensorTask, "Sensors", 4096, NULL, 1, NULL, 0);\n    xTaskCreatePinnedToCore(displayTask, "Display", 8192, NULL, 1, NULL, 1);\n}\n\nvoid loop() { vTaskDelay(pdMS_TO_TICKS(10000)); }'),
        ("Write ESP32 code to use LEDC for PWM fan speed control with a potentiometer.",
         '#include <Arduino.h>\n\nconst int POT_PIN = 34;\nconst int FAN_PIN = 16;\nconst int LEDC_CHANNEL = 0;\nconst int LEDC_FREQ = 25000;  // 25kHz for PC fans\nconst int LEDC_RESOLUTION = 8;\n\nvoid setup() {\n    Serial.begin(115200);\n    ledcSetup(LEDC_CHANNEL, LEDC_FREQ, LEDC_RESOLUTION);\n    ledcAttachPin(FAN_PIN, LEDC_CHANNEL);\n    analogReadResolution(12);\n}\n\nvoid loop() {\n    int potVal = analogRead(POT_PIN);\n    int duty = map(potVal, 0, 4095, 0, 255);\n    ledcWrite(LEDC_CHANNEL, duty);\n    Serial.printf("Pot: %d  Duty: %d/255\\n", potVal, duty);\n    delay(100);\n}'),
        ("Write ESP32 code to read a rotary encoder with interrupts and update a variable.",
         '#include <Arduino.h>\n\nconst int CLK_PIN = 25;\nconst int DT_PIN = 26;\nconst int SW_PIN = 27;\n\nvolatile int encoderPos = 0;\nvolatile bool buttonPressed = false;\n\nvoid IRAM_ATTR encoderISR() {\n    if (digitalRead(DT_PIN) == HIGH) {\n        encoderPos++;\n    } else {\n        encoderPos--;\n    }\n}\n\nvoid IRAM_ATTR buttonISR() {\n    buttonPressed = true;\n}\n\nvoid setup() {\n    Serial.begin(115200);\n    pinMode(CLK_PIN, INPUT_PULLUP);\n    pinMode(DT_PIN, INPUT_PULLUP);\n    pinMode(SW_PIN, INPUT_PULLUP);\n    attachInterrupt(digitalPinToInterrupt(CLK_PIN), encoderISR, FALLING);\n    attachInterrupt(digitalPinToInterrupt(SW_PIN), buttonISR, FALLING);\n}\n\nvoid loop() {\n    static int lastPos = 0;\n    noInterrupts();\n    int pos = encoderPos;\n    bool btn = buttonPressed;\n    buttonPressed = false;\n    interrupts();\n\n    if (pos != lastPos) {\n        Serial.printf("Position: %d\\n", pos);\n        lastPos = pos;\n    }\n    if (btn) {\n        Serial.println("Button pressed! Reset to 0");\n        noInterrupts();\n        encoderPos = 0;\n        interrupts();\n    }\n    delay(10);\n}'),
    ]
    for instr, code in esp32_patterns:
        records.append(make_msg(instr, code, {
            "source": "synthetic-esp32",
            "license": "CC0-1.0",
            "pattern": "esp32-specific",
            "domain_tag": "synthetic-embedded",
        }))

    # ── Pattern 7: FreeRTOS patterns (generic) ────────────────────────
    rtos_patterns = [
        ("Write FreeRTOS C++ code for a mutex-protected shared resource between two tasks.",
         '#include <Arduino.h>\n\nSemaphoreHandle_t xMutex;\nint sharedCounter = 0;\n\nvoid writerTask(void *param) {\n    for (;;) {\n        if (xSemaphoreTake(xMutex, pdMS_TO_TICKS(100)) == pdTRUE) {\n            sharedCounter++;\n            Serial.printf("[Writer] Counter: %d\\n", sharedCounter);\n            xSemaphoreGive(xMutex);\n        }\n        vTaskDelay(pdMS_TO_TICKS(500));\n    }\n}\n\nvoid readerTask(void *param) {\n    for (;;) {\n        if (xSemaphoreTake(xMutex, pdMS_TO_TICKS(100)) == pdTRUE) {\n            Serial.printf("[Reader] Counter: %d\\n", sharedCounter);\n            xSemaphoreGive(xMutex);\n        }\n        vTaskDelay(pdMS_TO_TICKS(300));\n    }\n}\n\nvoid setup() {\n    Serial.begin(115200);\n    xMutex = xSemaphoreCreateMutex();\n    xTaskCreate(writerTask, "Writer", 4096, NULL, 1, NULL);\n    xTaskCreate(readerTask, "Reader", 4096, NULL, 1, NULL);\n}\n\nvoid loop() { vTaskDelay(pdMS_TO_TICKS(60000)); }'),
        ("Write FreeRTOS code using a binary semaphore to signal between an ISR and a task.",
         '#include <Arduino.h>\n\nSemaphoreHandle_t xBinarySem;\nconst int BUTTON_PIN = 0;  // Boot button on ESP32\n\nvoid IRAM_ATTR buttonISR() {\n    BaseType_t xHigherPriorityTaskWoken = pdFALSE;\n    xSemaphoreGiveFromISR(xBinarySem, &xHigherPriorityTaskWoken);\n    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);\n}\n\nvoid processTask(void *param) {\n    for (;;) {\n        if (xSemaphoreTake(xBinarySem, portMAX_DELAY) == pdTRUE) {\n            Serial.println("Button event processed!");\n            // Do actual work here\n            vTaskDelay(pdMS_TO_TICKS(200));  // Debounce\n        }\n    }\n}\n\nvoid setup() {\n    Serial.begin(115200);\n    xBinarySem = xSemaphoreCreateBinary();\n    pinMode(BUTTON_PIN, INPUT_PULLUP);\n    attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), buttonISR, FALLING);\n    xTaskCreate(processTask, "Process", 4096, NULL, 2, NULL);\n}\n\nvoid loop() { vTaskDelay(pdMS_TO_TICKS(60000)); }'),
        ("Write FreeRTOS code with a software timer that periodically reads a sensor.",
         '#include <Arduino.h>\n\nTimerHandle_t sensorTimer;\nconst int SENSOR_PIN = 34;\n\nvoid sensorTimerCallback(TimerHandle_t xTimer) {\n    int val = analogRead(SENSOR_PIN);\n    Serial.printf("[Timer] Sensor: %d\\n", val);\n}\n\nvoid setup() {\n    Serial.begin(115200);\n    sensorTimer = xTimerCreate(\n        "SensorTimer",\n        pdMS_TO_TICKS(2000),  // 2 second period\n        pdTRUE,               // Auto-reload\n        NULL,\n        sensorTimerCallback\n    );\n    xTimerStart(sensorTimer, 0);\n    Serial.println("FreeRTOS timer started");\n}\n\nvoid loop() { vTaskDelay(pdMS_TO_TICKS(60000)); }'),
        ("Write FreeRTOS code using event groups to synchronize three tasks.",
         '#include <Arduino.h>\n\nEventGroupHandle_t xEventGroup;\n#define SENSOR_READY_BIT (1 << 0)\n#define WIFI_READY_BIT   (1 << 1)\n#define ALL_READY_BITS   (SENSOR_READY_BIT | WIFI_READY_BIT)\n\nvoid sensorTask(void *param) {\n    vTaskDelay(pdMS_TO_TICKS(1000));\n    Serial.println("Sensor initialized");\n    xEventGroupSetBits(xEventGroup, SENSOR_READY_BIT);\n    for (;;) {\n        int val = analogRead(34);\n        Serial.printf("Sensor: %d\\n", val);\n        vTaskDelay(pdMS_TO_TICKS(2000));\n    }\n}\n\nvoid wifiTask(void *param) {\n    vTaskDelay(pdMS_TO_TICKS(2000));\n    Serial.println("WiFi connected");\n    xEventGroupSetBits(xEventGroup, WIFI_READY_BIT);\n    for (;;) { vTaskDelay(pdMS_TO_TICKS(5000)); }\n}\n\nvoid mainTask(void *param) {\n    xEventGroupWaitBits(xEventGroup, ALL_READY_BITS, pdFALSE, pdTRUE, portMAX_DELAY);\n    Serial.println("All subsystems ready — starting main loop");\n    for (;;) {\n        Serial.println("Main processing...");\n        vTaskDelay(pdMS_TO_TICKS(1000));\n    }\n}\n\nvoid setup() {\n    Serial.begin(115200);\n    xEventGroup = xEventGroupCreate();\n    xTaskCreate(sensorTask, "Sensor", 4096, NULL, 1, NULL);\n    xTaskCreate(wifiTask, "WiFi", 8192, NULL, 1, NULL);\n    xTaskCreate(mainTask, "Main", 4096, NULL, 2, NULL);\n}\n\nvoid loop() { vTaskDelay(pdMS_TO_TICKS(60000)); }'),
    ]
    for instr, code in rtos_patterns:
        records.append(make_msg(instr, code, {
            "source": "synthetic-freertos",
            "license": "CC0-1.0",
            "pattern": "freertos",
            "domain_tag": "synthetic-embedded",
        }))

    # ── Pattern 8: Common embedded patterns (state machine, ring buffer, etc.)
    embedded_patterns = [
        ("Write a ring buffer implementation in C++ suitable for embedded systems (no dynamic allocation).",
         '#include <stdint.h>\n#include <string.h>\n\ntemplate<typename T, uint16_t SIZE>\nclass RingBuffer {\nprivate:\n    T buffer[SIZE];\n    volatile uint16_t head = 0;\n    volatile uint16_t tail = 0;\n    volatile uint16_t count = 0;\n\npublic:\n    bool push(const T &item) {\n        if (count >= SIZE) return false;\n        buffer[head] = item;\n        head = (head + 1) % SIZE;\n        count++;\n        return true;\n    }\n\n    bool pop(T &item) {\n        if (count == 0) return false;\n        item = buffer[tail];\n        tail = (tail + 1) % SIZE;\n        count--;\n        return true;\n    }\n\n    bool isEmpty() const { return count == 0; }\n    bool isFull() const { return count >= SIZE; }\n    uint16_t size() const { return count; }\n    uint16_t capacity() const { return SIZE; }\n};\n\n// Usage example with UART receive buffer\nRingBuffer<uint8_t, 256> uartRxBuffer;\n\n// Called from UART ISR\nvoid onUartRx(uint8_t byte) {\n    uartRxBuffer.push(byte);\n}\n\n// Called from main loop\nvoid processUart() {\n    uint8_t byte;\n    while (uartRxBuffer.pop(byte)) {\n        // Process received byte\n    }\n}'),
        ("Write a finite state machine (FSM) in C++ for an embedded traffic light controller.",
         '#include <Arduino.h>\n\nenum class TrafficState : uint8_t {\n    RED,\n    RED_YELLOW,\n    GREEN,\n    YELLOW\n};\n\nconst int RED_PIN = 2;\nconst int YELLOW_PIN = 3;\nconst int GREEN_PIN = 4;\n\nTrafficState currentState = TrafficState::RED;\nunsigned long stateEnterTime = 0;\n\nstruct StateConfig {\n    TrafficState state;\n    bool red, yellow, green;\n    unsigned long duration_ms;\n    TrafficState nextState;\n};\n\nconst StateConfig states[] = {\n    {TrafficState::RED,        true,  false, false, 5000, TrafficState::RED_YELLOW},\n    {TrafficState::RED_YELLOW, true,  true,  false, 1500, TrafficState::GREEN},\n    {TrafficState::GREEN,      false, false, true,  5000, TrafficState::YELLOW},\n    {TrafficState::YELLOW,     false, true,  false, 1500, TrafficState::RED},\n};\n\nconst StateConfig& getConfig(TrafficState s) {\n    for (const auto &cfg : states) {\n        if (cfg.state == s) return cfg;\n    }\n    return states[0];\n}\n\nvoid applyState(const StateConfig &cfg) {\n    digitalWrite(RED_PIN, cfg.red);\n    digitalWrite(YELLOW_PIN, cfg.yellow);\n    digitalWrite(GREEN_PIN, cfg.green);\n}\n\nvoid setup() {\n    Serial.begin(9600);\n    pinMode(RED_PIN, OUTPUT);\n    pinMode(YELLOW_PIN, OUTPUT);\n    pinMode(GREEN_PIN, OUTPUT);\n    applyState(getConfig(currentState));\n    stateEnterTime = millis();\n}\n\nvoid loop() {\n    const StateConfig &cfg = getConfig(currentState);\n    if (millis() - stateEnterTime >= cfg.duration_ms) {\n        currentState = cfg.nextState;\n        applyState(getConfig(currentState));\n        stateEnterTime = millis();\n        Serial.printf("State -> %d\\n", (int)currentState);\n    }\n}'),
        ("Write a CRC-8 calculation function in C for embedded firmware use.",
         '#include <stdint.h>\n\n// CRC-8 with polynomial 0x07 (CRC-8/CCITT)\nstatic const uint8_t crc8_table[256] = {\n    0x00, 0x07, 0x0E, 0x09, 0x1C, 0x1B, 0x12, 0x15,\n    0x38, 0x3F, 0x36, 0x31, 0x24, 0x23, 0x2A, 0x2D,\n    0x70, 0x77, 0x7E, 0x79, 0x6C, 0x6B, 0x62, 0x65,\n    0x48, 0x4F, 0x46, 0x41, 0x54, 0x53, 0x5A, 0x5D,\n    // ... (truncated for brevity, full 256-byte table in production)\n};\n\nuint8_t crc8(const uint8_t *data, uint16_t len) {\n    uint8_t crc = 0x00;\n    while (len--) {\n        crc = crc8_table[crc ^ *data++];\n    }\n    return crc;\n}\n\n// Bit-by-bit version (smaller code, slower)\nuint8_t crc8_slow(const uint8_t *data, uint16_t len) {\n    uint8_t crc = 0x00;\n    while (len--) {\n        crc ^= *data++;\n        for (uint8_t i = 0; i < 8; i++) {\n            if (crc & 0x80) {\n                crc = (crc << 1) ^ 0x07;\n            } else {\n                crc <<= 1;\n            }\n        }\n    }\n    return crc;\n}\n\n// Usage: validate a packet\nbool validatePacket(const uint8_t *packet, uint16_t len) {\n    // Last byte is CRC\n    uint8_t expected = packet[len - 1];\n    uint8_t computed = crc8(packet, len - 1);\n    return computed == expected;\n}'),
        ("Write a debounce class in C++ for embedded button handling with configurable delay.",
         '#include <Arduino.h>\n\nclass Debounce {\nprivate:\n    uint8_t pin;\n    bool lastState;\n    bool stableState;\n    unsigned long lastChangeTime;\n    unsigned long debounceDelay;\n    bool pullup;\n\npublic:\n    Debounce(uint8_t pin, unsigned long delay_ms = 50, bool pullup = true)\n        : pin(pin), debounceDelay(delay_ms), pullup(pullup) {\n        if (pullup) {\n            pinMode(pin, INPUT_PULLUP);\n        } else {\n            pinMode(pin, INPUT);\n        }\n        lastState = digitalRead(pin);\n        stableState = lastState;\n        lastChangeTime = millis();\n    }\n\n    bool update() {\n        bool reading = digitalRead(pin);\n        if (reading != lastState) {\n            lastChangeTime = millis();\n        }\n        lastState = reading;\n\n        if ((millis() - lastChangeTime) > debounceDelay) {\n            if (reading != stableState) {\n                stableState = reading;\n                return true;  // State changed\n            }\n        }\n        return false;\n    }\n\n    bool isPressed() const {\n        return pullup ? !stableState : stableState;\n    }\n\n    bool fell() { return update() && isPressed(); }\n    bool rose() { return update() && !isPressed(); }\n};\n\n// Usage\nDebounce btn1(2, 50);\nDebounce btn2(3, 30);\n\nvoid setup() {\n    Serial.begin(9600);\n}\n\nvoid loop() {\n    if (btn1.fell()) Serial.println("Button 1 pressed");\n    if (btn2.fell()) Serial.println("Button 2 pressed");\n}'),
        ("Write a fixed-point math library in C for embedded systems without FPU.",
         '#include <stdint.h>\n\n// Q16.16 fixed-point type\ntypedef int32_t fixed_t;\n\n#define FIXED_SHIFT 16\n#define FIXED_ONE   (1 << FIXED_SHIFT)\n#define FIXED_HALF  (1 << (FIXED_SHIFT - 1))\n\n// Conversions\nstatic inline fixed_t int_to_fixed(int x) { return (fixed_t)x << FIXED_SHIFT; }\nstatic inline int fixed_to_int(fixed_t x) { return (int)(x >> FIXED_SHIFT); }\nstatic inline fixed_t float_to_fixed(float x) { return (fixed_t)(x * FIXED_ONE); }\nstatic inline float fixed_to_float(fixed_t x) { return (float)x / FIXED_ONE; }\n\n// Arithmetic\nstatic inline fixed_t fixed_add(fixed_t a, fixed_t b) { return a + b; }\nstatic inline fixed_t fixed_sub(fixed_t a, fixed_t b) { return a - b; }\n\nstatic inline fixed_t fixed_mul(fixed_t a, fixed_t b) {\n    return (fixed_t)(((int64_t)a * b) >> FIXED_SHIFT);\n}\n\nstatic inline fixed_t fixed_div(fixed_t a, fixed_t b) {\n    return (fixed_t)(((int64_t)a << FIXED_SHIFT) / b);\n}\n\n// Linear interpolation\nstatic inline fixed_t fixed_lerp(fixed_t a, fixed_t b, fixed_t t) {\n    return fixed_add(a, fixed_mul(fixed_sub(b, a), t));\n}\n\n// Example: PID controller using fixed-point\ntypedef struct {\n    fixed_t kp, ki, kd;\n    fixed_t integral;\n    fixed_t prev_error;\n} PID;\n\nfixed_t pid_update(PID *pid, fixed_t setpoint, fixed_t measured) {\n    fixed_t error = fixed_sub(setpoint, measured);\n    pid->integral = fixed_add(pid->integral, error);\n    fixed_t derivative = fixed_sub(error, pid->prev_error);\n    pid->prev_error = error;\n\n    return fixed_add(\n        fixed_add(fixed_mul(pid->kp, error), fixed_mul(pid->ki, pid->integral)),\n        fixed_mul(pid->kd, derivative)\n    );\n}'),
        ("Write embedded C++ code for a simple command parser over UART (Serial).",
         '#include <Arduino.h>\n\nconst int BUF_SIZE = 64;\nchar cmdBuffer[BUF_SIZE];\nint cmdIndex = 0;\n\nvoid processCommand(const char *cmd) {\n    if (strcmp(cmd, "led on") == 0) {\n        digitalWrite(LED_BUILTIN, HIGH);\n        Serial.println("OK: LED on");\n    } else if (strcmp(cmd, "led off") == 0) {\n        digitalWrite(LED_BUILTIN, LOW);\n        Serial.println("OK: LED off");\n    } else if (strncmp(cmd, "pwm ", 4) == 0) {\n        int val = atoi(cmd + 4);\n        val = constrain(val, 0, 255);\n        analogWrite(9, val);\n        Serial.printf("OK: PWM=%d\\n", val);\n    } else if (strcmp(cmd, "adc") == 0) {\n        int val = analogRead(A0);\n        Serial.printf("ADC: %d\\n", val);\n    } else if (strcmp(cmd, "help") == 0) {\n        Serial.println("Commands: led on|off, pwm <0-255>, adc, help");\n    } else {\n        Serial.print("ERR: unknown cmd: ");\n        Serial.println(cmd);\n    }\n}\n\nvoid setup() {\n    Serial.begin(9600);\n    pinMode(LED_BUILTIN, OUTPUT);\n    pinMode(9, OUTPUT);\n    Serial.println("Ready. Type \'help\' for commands.");\n}\n\nvoid loop() {\n    while (Serial.available()) {\n        char c = Serial.read();\n        if (c == \'\\n\' || c == \'\\r\') {\n            if (cmdIndex > 0) {\n                cmdBuffer[cmdIndex] = \'\\0\';\n                processCommand(cmdBuffer);\n                cmdIndex = 0;\n            }\n        } else if (cmdIndex < BUF_SIZE - 1) {\n            cmdBuffer[cmdIndex++] = c;\n        }\n    }\n}'),
    ]
    for instr, code in embedded_patterns:
        records.append(make_msg(instr, code, {
            "source": "synthetic-embedded-pattern",
            "license": "CC0-1.0",
            "pattern": "embedded-common",
            "domain_tag": "synthetic-embedded",
        }))

    # Cap to target and shuffle
    rng.shuffle(records)
    if len(records) > target:
        records = records[:target]

    return records


def generate_synthetic_from_templates(rng: random.Random) -> list[dict]:
    """Generate synthetic embedded C++ instruction pairs from templates + combinatorial."""
    records: list[dict] = []
    # Original template-based records
    for tmpl in SYNTHETIC_TEMPLATES:
        for variant in tmpl["variants"]:
            instruction = tmpl["instruction"].format(**variant)
            code = tmpl["code"].format(**variant)
            provenance = {
                "source": "synthetic-template",
                "license": "CC0-1.0",
                "template_pattern": tmpl["pattern"],
                "domain_tag": "synthetic-embedded",
            }
            records.append(make_msg(instruction, code, provenance))

    # Add combinatorial synthetics
    combinatorial = generate_combinatorial_synthetics(rng, target=1200)
    records.extend(combinatorial)

    rng.shuffle(records)
    return records


def generate_synthetic_from_oshwa(rng: random.Random, max_records: int = 3000) -> list[dict]:
    """Read OSHWA data and generate synthetic embedded C++ instruction pairs."""
    if not OSHWA_PATH.exists():
        print(f"  OSHWA data not found at {OSHWA_PATH}")
        return []

    oshwa_records: list[dict] = []
    with open(OSHWA_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            oshwa_records.append(json.loads(line))

    # Extract hardware projects with useful descriptions
    hw_projects: list[dict] = []
    for rec in oshwa_records:
        if not rec.get("messages") or len(rec["messages"]) < 2:
            continue
        desc = rec["messages"][1]["content"]
        prov = rec.get("_provenance", {})
        # Only use projects that mention hardware interfaces
        desc_lower = desc.lower()
        has_interface = any(
            kw in desc_lower
            for kw in ["i2c", "spi", "uart", "serial", "gpio", "pwm", "adc",
                        "sensor", "motor", "led", "display", "oled", "lcd",
                        "wifi", "bluetooth", "ble", "usb", "can", "midi"]
        )
        if has_interface and len(desc) > 50:
            hw_projects.append({"desc": desc, "prov": prov})

    rng.shuffle(hw_projects)

    # Generate instruction pairs based on OSHWA descriptions
    # Each project gets 1 randomly chosen template → more diversity
    instruction_templates: list[str] = [
        "Write embedded C++ code for an Arduino-based implementation of this project: {desc_short}. Focus on the {interface} communication.",
        "Implement the core {interface} driver in C++ for this open hardware project: {desc_short}.",
        "Write a minimal embedded C++ firmware sketch for: {desc_short}. Include {interface} initialization and a main loop.",
        "Write C++ code for an ESP32 that implements: {desc_short}. Use {interface} for the hardware interface.",
        "Create an STM32-compatible C++ sketch for: {desc_short}. Initialize {interface} and handle the main control loop.",
        "Write Arduino C++ code with proper {interface} setup for: {desc_short}. Include error handling.",
    ]

    # Detect interface from description
    interface_map: dict[str, str] = {
        "i2c": "I2C", "spi": "SPI", "uart": "UART", "serial": "Serial",
        "gpio": "GPIO", "pwm": "PWM", "adc": "ADC",
        "wifi": "WiFi", "bluetooth": "Bluetooth", "ble": "BLE",
        "usb": "USB", "can": "CAN bus", "midi": "MIDI",
    }

    # Simple code templates keyed by interface
    code_templates: dict[str, str] = {
        "I2C": '#include <Arduino.h>\n#include <Wire.h>\n\n// {desc_short}\n\nvoid setup() {{\n    Serial.begin(9600);\n    Wire.begin();\n    Serial.println("I2C device ready");\n}}\n\nvoid loop() {{\n    Wire.beginTransmission(0x50);  // Device address\n    Wire.write(0x00);  // Register\n    Wire.endTransmission();\n    Wire.requestFrom(0x50, 2);\n    if (Wire.available() >= 2) {{\n        int16_t data = (Wire.read() << 8) | Wire.read();\n        Serial.print("Data: ");\n        Serial.println(data);\n    }}\n    delay(1000);\n}}',
        "SPI": '#include <Arduino.h>\n#include <SPI.h>\n\n// {desc_short}\nconst int CS_PIN = 10;\n\nvoid setup() {{\n    Serial.begin(9600);\n    SPI.begin();\n    pinMode(CS_PIN, OUTPUT);\n    digitalWrite(CS_PIN, HIGH);\n}}\n\nuint8_t spiTransfer(uint8_t reg, uint8_t val) {{\n    digitalWrite(CS_PIN, LOW);\n    SPI.transfer(reg);\n    uint8_t result = SPI.transfer(val);\n    digitalWrite(CS_PIN, HIGH);\n    return result;\n}}\n\nvoid loop() {{\n    uint8_t data = spiTransfer(0x00, 0x00);\n    Serial.print("SPI data: ");\n    Serial.println(data, HEX);\n    delay(500);\n}}',
        "Serial": '#include <Arduino.h>\n\n// {desc_short}\n\nvoid setup() {{\n    Serial.begin(115200);\n    Serial.println("Device initialized");\n}}\n\nvoid loop() {{\n    if (Serial.available() > 0) {{\n        String cmd = Serial.readStringUntil(\'\\n\');\n        cmd.trim();\n        Serial.print("Received: ");\n        Serial.println(cmd);\n        // Process command\n        if (cmd == "status") {{\n            Serial.println("OK");\n        }}\n    }}\n}}',
        "GPIO": '#include <Arduino.h>\n\n// {desc_short}\nconst int OUTPUT_PIN = 13;\nconst int INPUT_PIN = 2;\n\nvoid setup() {{\n    Serial.begin(9600);\n    pinMode(OUTPUT_PIN, OUTPUT);\n    pinMode(INPUT_PIN, INPUT_PULLUP);\n}}\n\nvoid loop() {{\n    bool state = digitalRead(INPUT_PIN);\n    digitalWrite(OUTPUT_PIN, !state);\n    Serial.print("Input: ");\n    Serial.println(state ? "HIGH" : "LOW");\n    delay(100);\n}}',
        "PWM": '#include <Arduino.h>\n\n// {desc_short}\nconst int PWM_PIN = 9;\n\nvoid setup() {{\n    Serial.begin(9600);\n    pinMode(PWM_PIN, OUTPUT);\n}}\n\nvoid loop() {{\n    for (int duty = 0; duty <= 255; duty += 5) {{\n        analogWrite(PWM_PIN, duty);\n        delay(20);\n    }}\n    for (int duty = 255; duty >= 0; duty -= 5) {{\n        analogWrite(PWM_PIN, duty);\n        delay(20);\n    }}\n}}',
        "ADC": '#include <Arduino.h>\n\n// {desc_short}\nconst int SENSOR_PIN = A0;\n\nvoid setup() {{\n    Serial.begin(9600);\n}}\n\nvoid loop() {{\n    int raw = analogRead(SENSOR_PIN);\n    float voltage = raw * (5.0 / 1023.0);\n    Serial.print("ADC: ");\n    Serial.print(raw);\n    Serial.print("  Voltage: ");\n    Serial.print(voltage, 2);\n    Serial.println("V");\n    delay(500);\n}}',
    }
    # Fallback for less common interfaces
    default_code = '#include <Arduino.h>\n\n// {desc_short}\n\nvoid setup() {{\n    Serial.begin(115200);\n    Serial.println("Open hardware device ready");\n    // TODO: Initialize {interface} interface\n}}\n\nvoid loop() {{\n    // Main control loop\n    Serial.println("Running...");\n    delay(1000);\n}}'

    # Additional code templates for more variety
    esp32_code_templates: dict[str, str] = {
        "I2C": '#include <Arduino.h>\n#include <Wire.h>\n\n// {desc_short}\n#define DEVICE_ADDR 0x50\n\nvoid setup() {{\n    Serial.begin(115200);\n    Wire.begin(21, 22);  // ESP32 I2C pins\n    Serial.println("ESP32 I2C device ready");\n}}\n\nvoid loop() {{\n    Wire.beginTransmission(DEVICE_ADDR);\n    Wire.write(0x00);\n    Wire.endTransmission(false);\n    Wire.requestFrom((uint8_t)DEVICE_ADDR, (uint8_t)2);\n    if (Wire.available() >= 2) {{\n        int16_t data = (Wire.read() << 8) | Wire.read();\n        Serial.printf("Data: %d\\n", data);\n    }}\n    delay(1000);\n}}',
        "SPI": '#include <Arduino.h>\n#include <SPI.h>\n\n// {desc_short}\nconst int CS_PIN = 5;\n\nvoid setup() {{\n    Serial.begin(115200);\n    SPI.begin(18, 19, 23, CS_PIN);  // ESP32 default SPI pins\n    pinMode(CS_PIN, OUTPUT);\n    digitalWrite(CS_PIN, HIGH);\n}}\n\nuint8_t spiTransfer(uint8_t reg, uint8_t val) {{\n    digitalWrite(CS_PIN, LOW);\n    SPI.transfer(reg);\n    uint8_t result = SPI.transfer(val);\n    digitalWrite(CS_PIN, HIGH);\n    return result;\n}}\n\nvoid loop() {{\n    uint8_t data = spiTransfer(0x00, 0x00);\n    Serial.printf("SPI data: 0x%02X\\n", data);\n    delay(500);\n}}',
        "Serial": '#include <Arduino.h>\n\n// {desc_short}\n\nvoid setup() {{\n    Serial.begin(115200);\n    Serial2.begin(9600, SERIAL_8N1, 16, 17);  // ESP32 UART2\n    Serial.println("ESP32 dual UART ready");\n}}\n\nvoid loop() {{\n    // Forward data between Serial and Serial2\n    if (Serial2.available()) {{\n        String data = Serial2.readStringUntil(\'\\n\');\n        Serial.printf("Received: %s\\n", data.c_str());\n    }}\n    if (Serial.available()) {{\n        String cmd = Serial.readStringUntil(\'\\n\');\n        Serial2.println(cmd);\n    }}\n}}',
        "GPIO": '#include <Arduino.h>\n\n// {desc_short}\nconst int OUTPUT_PIN = 2;  // Built-in LED\nconst int INPUT_PIN = 0;   // Boot button\n\nvoid setup() {{\n    Serial.begin(115200);\n    pinMode(OUTPUT_PIN, OUTPUT);\n    pinMode(INPUT_PIN, INPUT_PULLUP);\n}}\n\nvoid loop() {{\n    bool state = digitalRead(INPUT_PIN);\n    digitalWrite(OUTPUT_PIN, !state);\n    Serial.printf("Input: %s\\n", state ? "HIGH" : "LOW");\n    delay(100);\n}}',
        "PWM": '#include <Arduino.h>\n\n// {desc_short}\nconst int PWM_PIN = 16;\nconst int LEDC_CHANNEL = 0;\n\nvoid setup() {{\n    Serial.begin(115200);\n    ledcSetup(LEDC_CHANNEL, 5000, 8);  // 5kHz, 8-bit\n    ledcAttachPin(PWM_PIN, LEDC_CHANNEL);\n}}\n\nvoid loop() {{\n    for (int duty = 0; duty <= 255; duty += 5) {{\n        ledcWrite(LEDC_CHANNEL, duty);\n        delay(20);\n    }}\n    for (int duty = 255; duty >= 0; duty -= 5) {{\n        ledcWrite(LEDC_CHANNEL, duty);\n        delay(20);\n    }}\n}}',
    }

    stm32_code_templates: dict[str, str] = {
        "I2C": '#include "stm32f4xx_hal.h"\n#include <stdio.h>\n\n// {desc_short}\nI2C_HandleTypeDef hi2c1;\nUART_HandleTypeDef huart2;\n#define DEVICE_ADDR (0x50 << 1)\n\nvoid SystemClock_Config(void);\n\nint main(void) {{\n    HAL_Init();\n    SystemClock_Config();\n    // I2C and UART init assumed via MX_xxx_Init()\n\n    uint8_t data[2];\n    char buf[64];\n    while (1) {{\n        HAL_I2C_Mem_Read(&hi2c1, DEVICE_ADDR, 0x00, I2C_MEMADD_SIZE_8BIT, data, 2, 100);\n        int16_t val = (data[0] << 8) | data[1];\n        int len = snprintf(buf, sizeof(buf), "I2C data: %d\\r\\n", val);\n        HAL_UART_Transmit(&huart2, (uint8_t*)buf, len, 100);\n        HAL_Delay(1000);\n    }}\n}}',
        "GPIO": '#include "stm32f4xx_hal.h"\n\n// {desc_short}\n\nvoid SystemClock_Config(void);\nstatic void MX_GPIO_Init(void);\n\nint main(void) {{\n    HAL_Init();\n    SystemClock_Config();\n    MX_GPIO_Init();\n\n    while (1) {{\n        if (HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_13) == GPIO_PIN_RESET) {{\n            HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);\n            HAL_Delay(200);  // Debounce\n        }}\n    }}\n}}\n\nstatic void MX_GPIO_Init(void) {{\n    __HAL_RCC_GPIOA_CLK_ENABLE();\n    __HAL_RCC_GPIOC_CLK_ENABLE();\n    GPIO_InitTypeDef gpio = {{0}};\n    gpio.Pin = GPIO_PIN_5;\n    gpio.Mode = GPIO_MODE_OUTPUT_PP;\n    HAL_GPIO_Init(GPIOA, &gpio);\n    gpio.Pin = GPIO_PIN_13;\n    gpio.Mode = GPIO_MODE_INPUT;\n    gpio.Pull = GPIO_PULLUP;\n    HAL_GPIO_Init(GPIOC, &gpio);\n}}',
    }

    records: list[dict] = []
    for proj in hw_projects:
        desc = proj["desc"]
        # Clean HTML entities
        import html as _html
        desc = _html.unescape(desc)
        desc_short = desc[:120].replace('"', "'").replace("\n", " ").replace("\r", "")

        # Detect interface
        desc_lower = desc.lower()
        detected_interface = "GPIO"
        for key, label in interface_map.items():
            if key in desc_lower:
                detected_interface = label
                break

        # Generate 1 Arduino variant
        tmpl_idx = rng.randint(0, 2)  # First 3 templates are Arduino-flavored
        instruction = instruction_templates[tmpl_idx].format(
            desc_short=desc_short, interface=detected_interface
        )
        code_tmpl = code_templates.get(detected_interface, default_code)
        code = code_tmpl.format(desc_short=desc_short, interface=detected_interface)
        provenance = {
            "source": "oshwa-synthetic",
            "license": proj["prov"].get("hardware_license", "Open"),
            "oshwa_uid": proj["prov"].get("uid", "unknown"),
            "domain_tag": "synthetic-embedded",
        }
        records.append(make_msg(instruction, code, provenance))

        # Generate 1 ESP32 variant (50% chance)
        if rng.random() < 0.5 and detected_interface in esp32_code_templates:
            tmpl_idx = 3  # ESP32 template
            instruction = instruction_templates[tmpl_idx].format(
                desc_short=desc_short, interface=detected_interface
            )
            code = esp32_code_templates[detected_interface].format(
                desc_short=desc_short, interface=detected_interface
            )
            provenance = {
                "source": "oshwa-synthetic-esp32",
                "license": proj["prov"].get("hardware_license", "Open"),
                "oshwa_uid": proj["prov"].get("uid", "unknown"),
                "domain_tag": "synthetic-embedded",
            }
            records.append(make_msg(instruction, code, provenance))

        # Generate 1 STM32 variant (30% chance)
        if rng.random() < 0.3 and detected_interface in stm32_code_templates:
            tmpl_idx = 4  # STM32 template
            instruction = instruction_templates[tmpl_idx].format(
                desc_short=desc_short, interface=detected_interface
            )
            code = stm32_code_templates[detected_interface].format(
                desc_short=desc_short, interface=detected_interface
            )
            provenance = {
                "source": "oshwa-synthetic-stm32",
                "license": proj["prov"].get("hardware_license", "Open"),
                "oshwa_uid": proj["prov"].get("uid", "unknown"),
                "domain_tag": "synthetic-embedded",
            }
            records.append(make_msg(instruction, code, provenance))

    rng.shuffle(records)
    if len(records) > max_records:
        records = records[:max_records]
    return records


def load_source_d(rng: random.Random) -> tuple[list[dict], dict]:
    """Generate synthetic embedded C++ from OSHWA data + templates."""
    print(f"\n{'='*60}")
    print(f"Source D: Synthetic embedded C++ (templates + OSHWA)")
    print(f"{'='*60}")

    template_records = generate_synthetic_from_templates(rng)
    print(f"  Template-generated: {len(template_records)}")

    oshwa_records = generate_synthetic_from_oshwa(rng, max_records=3000)
    print(f"  OSHWA-derived: {len(oshwa_records)}")

    records = template_records + oshwa_records
    rng.shuffle(records)

    stats = {"total": len(records), "pass": len(records), "contaminated": 0}
    print(f"  Total synthetic: {len(records)}")
    return records, stats


# ─── Source E: Generic C++ fallback ──────────────────────────────────────────

def load_source_e(ds, exclude_ids: set[str]) -> tuple[list[dict], dict]:
    """Generic C++ (>=2 markers) as fallback padding."""
    source_id = "iamtarun/code_instructions_120k_alpaca"
    license_ = "Apache-2.0"
    print(f"\n{'='*60}")
    print(f"Source E: {source_id} — generic C++ fallback")
    print(f"{'='*60}")

    records: list[dict] = []
    stats = {"total": len(ds), "pass": 0, "contaminated": 0, "already_used": 0}

    for idx, row in enumerate(ds):
        if str(idx) in exclude_ids:
            stats["already_used"] += 1
            continue

        user_text, output = extract_alpaca_text(row)
        if not user_text or not output:
            continue

        combined = f"{user_text}\n{output}"

        if is_contaminated(combined):
            stats["contaminated"] += 1
            continue

        cpp_count, _ = count_cpp_markers(combined)
        if cpp_count < 2:
            continue

        stats["pass"] += 1
        provenance = {
            "source": source_id,
            "license": license_,
            "record_id": str(idx),
            "domain_tag": "embedded-generic-cpp",
        }
        records.append(make_msg(user_text, output, provenance))

    print(f"  Generic C++ pass (>=2 markers): {stats['pass']}")
    print(f"  Already used in A/C: {stats['already_used']}")
    print(f"  Contaminated rejected: {stats['contaminated']}")
    return records, stats


# ─── Save / Manifest / Stats ────────────────────────────────────────────────

def save_split(records: list[dict], domain: str = "cpp") -> tuple[int, int]:
    """Shuffle, split, and write train/valid JSONL."""
    rng = random.Random(SEED)
    rng.shuffle(records)

    n_val = max(1, round(len(records) * VALID_RATIO))
    train, valid = records[n_val:], records[:n_val]

    d = OUT / domain
    d.mkdir(parents=True, exist_ok=True)

    for name, data in [("train.jsonl", train), ("valid.jsonl", valid)]:
        with open(d / name, "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  -> {domain}: {len(train)} train / {len(valid)} valid")
    return len(train), len(valid)


def update_manifest(
    hf_id: str,
    license_: str,
    n_source: int,
    n_used: int,
    n_train: int,
    n_valid: int,
    notes: str,
) -> None:
    """Update the cpp entry in MANIFEST_niche.json."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    else:
        manifest = []

    # Remove existing cpp entry
    manifest = [e for e in manifest if e.get("domain") != "cpp"]

    manifest.append(
        {
            "domain": "cpp",
            "hf_id": hf_id,
            "license": license_,
            "n_source": n_source,
            "n_used": n_used,
            "n_train": n_train,
            "n_valid": n_valid,
            "access_date": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
        }
    )

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"  -> MANIFEST_niche.json updated (cpp entry)")


def print_embedded_keyword_stats(records: list[dict]) -> None:
    """Print top 20 embedded keyword distribution."""
    keyword_counts: Counter = Counter()
    records_with_embedded = 0

    for rec in records:
        combined = rec["messages"][0]["content"] + "\n" + rec["messages"][1]["content"]
        found_any = False
        for kw in ALL_EMBEDDED_KEYWORDS:
            if kw.lower() in combined.lower():
                keyword_counts[kw] += 1
                found_any = True
        # Check bit manipulation
        if any(p.search(combined) for p in BIT_MANIP_PATTERNS):
            keyword_counts["bit_manipulation"] += 1
            found_any = True
        if found_any:
            records_with_embedded += 1

    pct_embedded = records_with_embedded / len(records) * 100 if records else 0

    print(f"\n{'='*60}")
    print(f"Embedded Keyword Coverage: {records_with_embedded}/{len(records)} ({pct_embedded:.1f}%)")
    print(f"{'='*60}")
    print(f"\nTop 20 most common embedded keywords:")
    for kw, count in keyword_counts.most_common(20):
        pct = count / len(records) * 100
        print(f"  {kw:25s}: {count:5d} ({pct:5.1f}%)")


def print_domain_tag_stats(records: list[dict]) -> None:
    """Print distribution of domain_tag values."""
    tag_counts: Counter = Counter()
    for rec in records:
        tag = rec["_provenance"].get("domain_tag", "unknown")
        tag_counts[tag] += 1

    print(f"\n{'='*60}")
    print(f"Domain Tag Distribution:")
    print(f"{'='*60}")
    for tag, count in tag_counts.most_common():
        pct = count / len(records) * 100
        print(f"  {tag:30s}: {count:5d} ({pct:5.1f}%)")


def print_samples(records: list[dict], n: int = 5) -> None:
    """Print sample records for quality validation."""
    print(f"\n{'='*60}")
    print(f"Sample records ({n}):")
    print(f"{'='*60}")
    for i, rec in enumerate(records[:n]):
        user = rec["messages"][0]["content"]
        assistant = rec["messages"][1]["content"]
        print(f"\n--- Sample {i+1} [{rec['_provenance'].get('domain_tag', '?')}] ---")
        print(f"USER: {user[:200]}")
        print(f"ASSISTANT: {assistant[:400]}")
        print(f"PROVENANCE: {rec['_provenance']}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = random.Random(SEED)
    all_records: list[dict] = []
    source_stats: list[dict] = []
    source_names: list[str] = []

    # Load iamtarun once, reuse for sources A, C, E
    print("Loading iamtarun/code_instructions_120k_alpaca ...")
    ds_iamtarun = load_dataset("iamtarun/code_instructions_120k_alpaca", split="train")
    print(f"  Raw records: {len(ds_iamtarun)}")

    # --- Source A: Embedded MCU from iamtarun ---
    records_a, stats_a = load_source_a(ds_iamtarun)
    # Track which iamtarun record_ids are already used
    used_iamtarun_ids: set[str] = {
        r["_provenance"]["record_id"] for r in records_a
    }
    all_records.extend(records_a)
    source_stats.append({"label": "A (iamtarun embedded)", **stats_a})
    source_names.append("iamtarun/code_instructions_120k_alpaca")
    print(f"  Running total: {len(all_records)}")

    # --- Source B: Embedded MCU from bigcode ---
    records_b, stats_b = load_source_b()
    all_records.extend(records_b)
    source_stats.append({"label": "B (bigcode embedded)", **stats_b})
    source_names.append("bigcode/self-oss-instruct-sc2-exec-filter-50k")
    print(f"  Running total: {len(all_records)}")

    # --- Source C: Hardware-flavored C++ from iamtarun ---
    records_c, stats_c = load_source_c(ds_iamtarun, used_iamtarun_ids)
    used_iamtarun_ids.update(
        r["_provenance"]["record_id"] for r in records_c
        if r["_provenance"]["source"] == "iamtarun/code_instructions_120k_alpaca"
    )
    all_records.extend(records_c)
    source_stats.append({"label": "C (iamtarun hw-flavored)", **stats_c})
    print(f"  Running total: {len(all_records)}")

    # --- Source D: Synthetic ---
    records_d, stats_d = load_source_d(rng)
    all_records.extend(records_d)
    source_stats.append({"label": "D (synthetic embedded)", **stats_d})
    source_names.append("OSHWA+synthetic-templates")
    print(f"  Running total: {len(all_records)}")

    # --- Source E: Generic C++ (always load for mix, even if embedded is sufficient) ---
    embedded_count = sum(
        1 for r in all_records
        if r["_provenance"].get("domain_tag") in ("embedded-mcu", "synthetic-embedded")
    )
    print(f"\n  Embedded-specific records so far: {embedded_count}")

    records_e, stats_e = load_source_e(ds_iamtarun, used_iamtarun_ids)
    all_records.extend(records_e)
    source_stats.append({"label": "E (generic C++ fallback)", **stats_e})
    print(f"  Running total: {len(all_records)}")

    # --- Aggregation ---
    print(f"\n{'='*60}")
    print("Aggregation summary:")
    print(f"{'='*60}")
    print(f"  Total collected: {len(all_records)}")
    print(f"  Target: {TARGET}")

    if len(all_records) > TARGET:
        # Target: ~70% embedded, ~30% generic C++ (useful for embedded devs)
        EMBEDDED_RATIO = 0.70
        embedded_target = int(TARGET * EMBEDDED_RATIO)
        generic_target = TARGET - embedded_target

        embedded_recs = [
            r for r in all_records
            if r["_provenance"].get("domain_tag") in ("embedded-mcu", "synthetic-embedded")
        ]
        generic_recs = [
            r for r in all_records
            if r["_provenance"].get("domain_tag") == "embedded-generic-cpp"
        ]

        # Cap embedded to target ratio
        if len(embedded_recs) > embedded_target:
            embedded_recs = rng.sample(embedded_recs, embedded_target)
        actual_embedded = len(embedded_recs)

        # Fill the rest with generic C++
        needed_generic = TARGET - actual_embedded
        if len(generic_recs) > needed_generic:
            generic_recs = rng.sample(generic_recs, needed_generic)
        all_records = embedded_recs + generic_recs

        print(f"  Capped to {len(all_records)} ({actual_embedded} embedded + {len(generic_recs)} generic)")
    elif len(all_records) < TARGET:
        print(f"  WARNING: Only {len(all_records)} records, below target {TARGET}")

    n_used = len(all_records)

    # Save
    n_train, n_valid = save_split(all_records)

    # Update manifest
    all_sources = list(dict.fromkeys(source_names))  # deduplicate preserving order
    hf_id = "+".join(all_sources)
    update_manifest(
        hf_id=hf_id,
        license_="Apache-2.0+Open",
        n_source=-1,
        n_used=n_used,
        n_train=n_train,
        n_valid=n_valid,
        notes="Embedded/MCU-focused C++ (GPIO, UART, SPI, I2C, AVR, STM32, ESP32, Arduino, FreeRTOS); 60%+ MCU content",
    )

    # Stats
    print_domain_tag_stats(all_records)
    print_embedded_keyword_stats(all_records)
    print_samples(all_records)

    # Final report
    print(f"\n{'='*60}")
    print("FINAL REPORT:")
    print(f"{'='*60}")
    for s in source_stats:
        label = s["label"]
        passed = s.get("pass", 0)
        total = s.get("total", 0)
        contaminated = s.get("contaminated", 0)
        print(f"  {label}: {total} raw -> {passed} pass, {contaminated} contaminated")

    # Embedded coverage
    final_embedded = sum(
        1 for r in all_records
        if r["_provenance"].get("domain_tag") in ("embedded-mcu", "synthetic-embedded")
    )
    pct = final_embedded / len(all_records) * 100 if all_records else 0
    print(f"\n  Embedded MCU/synthetic: {final_embedded}/{len(all_records)} ({pct:.1f}%)")
    print(f"  Generic C++ padding: {len(all_records) - final_embedded}/{len(all_records)} ({100-pct:.1f}%)")
    print(f"  Combined: {n_used} used -> {n_train} train / {n_valid} valid")
    print(f"  Sources: {hf_id}")

    if pct < 60:
        print(f"\n  WARNING: Embedded coverage {pct:.1f}% is below 60% target")
    else:
        print(f"\n  PASS: Embedded coverage {pct:.1f}% meets 60% target")


if __name__ == "__main__":
    main()
