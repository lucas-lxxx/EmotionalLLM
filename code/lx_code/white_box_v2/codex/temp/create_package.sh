#!/bin/bash
# Package white-box attack code for distribution

PACKAGE_NAME="white_box_v2"
PACKAGE_DIR="/data1/lixiang/lx_code/white_box_v2/codex"
OUTPUT_FILE="${PACKAGE_DIR}/${PACKAGE_NAME}.zip"

echo "Packaging White-Box Attack Code..."
echo "=================================="

# Create temporary directory for packaging
TEMP_DIR=$(mktemp -d)
TARGET_DIR="${TEMP_DIR}/${PACKAGE_NAME}"
mkdir -p "${TARGET_DIR}"

echo "Copying core files..."

# Core attack files
cp "${PACKAGE_DIR}/attack_core.py" "${TARGET_DIR}/"
cp "${PACKAGE_DIR}/opens2s_io.py" "${TARGET_DIR}/"
cp "${PACKAGE_DIR}/config.py" "${TARGET_DIR}/"
cp "${PACKAGE_DIR}/run_attack.py" "${TARGET_DIR}/"
cp "${PACKAGE_DIR}/eval_metrics.py" "${TARGET_DIR}/"

# Documentation
cp "${PACKAGE_DIR}/PACKAGE_README.md" "${TARGET_DIR}/README.md"
cp "${PACKAGE_DIR}/RUN_GUIDE.md" "${TARGET_DIR}/"
cp "${PACKAGE_DIR}/FINDINGS.md" "${TARGET_DIR}/"

# Examples and utilities
cp "${PACKAGE_DIR}/sample_list.txt" "${TARGET_DIR}/"
cp "${PACKAGE_DIR}/quick_start.sh" "${TARGET_DIR}/"

echo "Creating archive..."
cd "${TEMP_DIR}"
zip -r "${OUTPUT_FILE}" "${PACKAGE_NAME}/"

# Cleanup
rm -rf "${TEMP_DIR}"

echo ""
echo "Package created successfully!"
echo "Location: ${OUTPUT_FILE}"
echo ""
echo "Contents:"
unzip -l "${OUTPUT_FILE}"

echo ""
echo "Package size:"
ls -lh "${OUTPUT_FILE}"
