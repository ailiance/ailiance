# EU-KIKI VLM Pipeline — Compliance Audit Trail

Generated: 2026-05-02T23:39:58.224859+00:00
Pipeline version: vlm-poc-0.1

## Legal Framework

- **EU Digital Single Market Directive, Article 4**: Text and Data Mining
  for research purposes is permitted when lawful access is available
  and the rights holder has not expressly reserved TDM rights.
- **EU AI Act (Regulation 2024/1689)**: Training data provenance must be
  documented and auditable. Article 53 requires transparency for GPAI models.
- **DSM Art.4 TDM applicability**: All sources used provide freely available
  technical documentation without login requirements. No TDM opt-out detected.

## Robots.txt Verification

| Source | Status | TDM Opt-Out | Checked At |
|--------|--------|-------------|------------|
| ST Application Notes | ALLOWED | No | 2026-05-02T23:37:04 |
| Espressif Documentation | ALLOWED | No | 2026-05-02T23:39:35 |

## PDFs Downloaded

### ST Application Notes

- **Legal basis**: DSM_ART4_TDM
- **Downloaded**: 2 / 2

| File | SHA-256 | Size | HTTP | Date |
|------|---------|------|------|------|
| `an4488-getting-started-with-stm32f4xxxx-mcu-hardwa` | `sha256:7704a013f43fc4b5d8...` | 1220KB | 200 | 2026-05-02 |
| `an2867-oscillator-design-guide-for-stm8afals-stm32` | `sha256:1c92e22e215ca66a15...` | 3132KB | 200 | 2026-05-02 |

### Espressif Documentation

- **Legal basis**: DSM_ART4_TDM
- **Downloaded**: 3 / 3

| File | SHA-256 | Size | HTTP | Date |
|------|---------|------|------|------|
| `esp32_datasheet_en.pdf` | `sha256:6fdff42cce00775643...` | 966KB | 200 | 2026-05-02 |
| `esp32-s3_datasheet_en.pdf` | `sha256:2d5a7cb7fd559d8d97...` | 1072KB | 200 | 2026-05-02 |
| `esp32_hardware_design_guidelines_en.pdf` | `sha256:259d7b566bbb8ef13c...` | 14KB | 200 | 2026-05-02 |

## Page Extraction

- **Total pages extracted**: 277
- **Image DPI**: 200
- **Classification method**: drawing count heuristic

| Page Type | Count | Classification Criteria |
|-----------|-------|------------------------|
| diagram | 60 | drawings > 5 and < 20 |
| schematic | 73 | drawings >= 20, not mostly lines |
| sparse | 3 | does not match other categories |
| table | 94 | drawings >= 20, >60% are lines |
| text | 47 | text >= 500 chars, few drawings |

### Per-PDF Breakdown

- **an2867-oscillator-design-guide-for-stm8afals-stm32-mcus-and-mpus-stmicroelectronics.pdf**: 60 pages (diagram:30, schematic:13, table:17)
- **an4488-getting-started-with-stm32f4xxxx-mcu-hardware-development-stmicroelectronics.pdf**: 50 pages (diagram:17, schematic:13, table:20)
- **esp32-s3_datasheet_en.pdf**: 87 pages (diagram:8, schematic:25, sparse:2, table:23, text:29)
- **esp32_datasheet_en.pdf**: 78 pages (diagram:4, schematic:22, sparse:1, table:33, text:18)
- **esp32_hardware_design_guidelines_en.pdf**: 2 pages (diagram:1, table:1)

## VLM Training Data

- **Total training pairs**: 924

| Page Type | Training Pairs |
|-----------|---------------|
| diagram | 180 |
| schematic | 365 |
| sparse | 3 |
| table | 282 |
| text | 94 |

### Sample Training Pair

- **Image**: `vlm-images/st_application_notes/an2867-oscillator-design-guide-for-stm8afals-stm32-mcus-and-mpus-stmicroelectronics/page_000.png`
- **Page type**: table
- **Question**: Extract and describe the data shown in this table.
- **Answer** (first 200 chars): The technical data shown includes:

February 2026
AN2867 Rev 24
1/60
1
AN2867
Application note
Guidelines for oscillator design on STM8AF/AL/S 
 and STM32 MCUs/MPUs
 
Introduction
Many designers know ...
- **Source**: ST Application Notes
- **Legal basis**: DSM_ART4_TDM

## Summary

- **Total PDFs downloaded**: 5
- **Total pages extracted**: 277
- **Schematic/diagram pages**: 133
- **Total VLM training pairs**: 924
- **Report date**: 2026-05-02
- **All sources verified**: no TDM opt-out detected
