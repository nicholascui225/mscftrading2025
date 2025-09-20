#!/usr/bin/env python3
"""
Log Viewer for Volatility Trading Strategy
View and analyze log files from different cases
"""

import os
import glob
import datetime
from pathlib import Path

def list_log_files():
    """List all available log files"""
    logs_dir = Path('logs')
    if not logs_dir.exists():
        print("No logs directory found. Run the strategy first to generate logs.")
        return
    
    # Get all log files
    case_logs = list(logs_dir.glob('volatility_case_*.log'))
    summary_logs = list(logs_dir.glob('volatility_summary_*.log'))
    
    print("📁 Available Log Files:")
    print("=" * 60)
    
    if case_logs:
        print(f"\n📊 Case Logs ({len(case_logs)} files):")
        for log_file in sorted(case_logs):
            file_size = log_file.stat().st_size
            mod_time = datetime.datetime.fromtimestamp(log_file.stat().st_mtime)
            print(f"  📄 {log_file.name}")
            print(f"     Size: {file_size:,} bytes | Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if summary_logs:
        print(f"\n📈 Summary Logs ({len(summary_logs)} files):")
        for log_file in sorted(summary_logs):
            file_size = log_file.stat().st_size
            mod_time = datetime.datetime.fromtimestamp(log_file.stat().st_mtime)
            print(f"  📄 {log_file.name}")
            print(f"     Size: {file_size:,} bytes | Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")

def show_case_summary(case_number=None):
    """Show summary of a specific case or all cases"""
    logs_dir = Path('logs')
    if not logs_dir.exists():
        print("No logs directory found.")
        return
    
    summary_files = list(logs_dir.glob('volatility_summary_*.log'))
    if not summary_files:
        print("No summary files found.")
        return
    
    # Get the most recent summary file
    latest_summary = max(summary_files, key=lambda x: x.stat().st_mtime)
    
    print(f"📈 Case Summary from {latest_summary.name}:")
    print("=" * 60)
    
    with open(latest_summary, 'r') as f:
        content = f.read()
        if case_number:
            # Show specific case
            lines = content.split('\n')
            in_case = False
            for line in lines:
                if f"CASE #{case_number} SUMMARY" in line:
                    in_case = True
                elif in_case and line.startswith("="):
                    break
                elif in_case:
                    print(line)
        else:
            # Show all cases
            print(content)

def show_latest_case_log():
    """Show the latest case log file"""
    logs_dir = Path('logs')
    if not logs_dir.exists():
        print("No logs directory found.")
        return
    
    case_logs = list(logs_dir.glob('volatility_case_*.log'))
    if not case_logs:
        print("No case logs found.")
        return
    
    # Get the most recent case log
    latest_log = max(case_logs, key=lambda x: x.stat().st_mtime)
    
    print(f"📄 Latest Case Log: {latest_log.name}")
    print("=" * 60)
    
    # Show last 50 lines
    with open(latest_log, 'r') as f:
        lines = f.readlines()
        for line in lines[-50:]:
            print(line.rstrip())

def main():
    """Main function"""
    print("🔍 Volatility Trading Log Viewer")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. List all log files")
        print("2. Show case summary")
        print("3. Show latest case log")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            list_log_files()
        elif choice == '2':
            case_num = input("Enter case number (or press Enter for all): ").strip()
            if case_num:
                try:
                    show_case_summary(int(case_num))
                except ValueError:
                    print("Invalid case number.")
            else:
                show_case_summary()
        elif choice == '3':
            show_latest_case_log()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

