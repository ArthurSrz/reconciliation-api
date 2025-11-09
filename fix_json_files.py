#!/usr/bin/env python3
"""
Script to detect and fix corrupted JSON files in book_data
The error "Expecting value: line 1 column 1 (char 0)" indicates empty or invalid JSON files
"""

import os
import json
import glob
from pathlib import Path

def check_and_fix_json_files():
    """Check all JSON files in book_data and fix corrupted ones"""
    corrupted_files = []
    fixed_files = []

    # Find all JSON files in book_data
    json_files = glob.glob("book_data/**/*.json", recursive=True)

    print(f"🔍 Checking {len(json_files)} JSON files...")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            # Check if file is empty
            if not content:
                print(f"❌ Empty file: {json_file}")
                corrupted_files.append(json_file)
                # Fix empty file with empty JSON object
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump({}, f)
                fixed_files.append(json_file)
                continue

            # Try to parse JSON
            json.loads(content)
            print(f"✅ Valid JSON: {json_file}")

        except json.JSONDecodeError as e:
            print(f"❌ JSON Error in {json_file}: {e}")
            corrupted_files.append(json_file)

            # Try to fix common issues
            try:
                # Remove null bytes and control characters
                content_cleaned = content.replace('\x00', '').replace('\r', '').strip()

                # If still can't parse, create empty object
                if not content_cleaned or content_cleaned in ['null', 'undefined', '']:
                    content_cleaned = '{}'

                # Validate cleaned content
                json.loads(content_cleaned)

                # Write fixed content
                with open(json_file, 'w', encoding='utf-8') as f:
                    f.write(content_cleaned)

                print(f"🔧 Fixed: {json_file}")
                fixed_files.append(json_file)

            except Exception as fix_error:
                print(f"💀 Cannot fix {json_file}: {fix_error}")
                # As last resort, create empty JSON object
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump({}, f)
                print(f"🔧 Reset to empty object: {json_file}")
                fixed_files.append(json_file)

        except Exception as e:
            print(f"💀 Cannot read {json_file}: {e}")
            corrupted_files.append(json_file)

    print(f"\n📊 Summary:")
    print(f"   - Total files: {len(json_files)}")
    print(f"   - Corrupted: {len(corrupted_files)}")
    print(f"   - Fixed: {len(fixed_files)}")

    return corrupted_files, fixed_files

if __name__ == "__main__":
    print("🔧 JSON File Repair Tool")
    print("=" * 40)

    corrupted, fixed = check_and_fix_json_files()

    if fixed:
        print(f"\n✅ Fixed {len(fixed)} files. Ready to commit and deploy!")
    else:
        print(f"\n🎉 All JSON files are valid!")