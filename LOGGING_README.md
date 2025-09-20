# 📊 Volatility Trading Logging System

## Overview
The volatility trading strategy now creates separate log files for each trading case, making it easy to track performance across different sessions and analyze individual case results.

## 📁 Log File Structure

### Directory Structure
```
logs/
├── volatility_case_1_20250919_003745_tick70.log    # Case 1 detailed log
├── volatility_case_2_20250919_004230_tick0.log     # Case 2 detailed log
├── volatility_summary_20250919.log                  # Daily summary of all cases
└── volatility_trading_20250919_003700.log          # Initial startup log
```

### Log File Types

#### 1. **Case Logs** (`volatility_case_N_timestamp_tickX.log`)
- **Purpose**: Detailed log for each individual trading case
- **Content**: All trading decisions, orders, hedging, performance metrics
- **Naming**: `volatility_case_{case_number}_{timestamp}_tick{tick}.log`

#### 2. **Summary Logs** (`volatility_summary_YYYYMMDD.log`)
- **Purpose**: Daily summary of all cases run
- **Content**: Case summaries with key metrics
- **Naming**: `volatility_summary_{date}.log`

#### 3. **Initial Logs** (`volatility_trading_timestamp.log`)
- **Purpose**: Initial startup and case detection
- **Content**: Case detection, waiting for cases to start
- **Naming**: `volatility_trading_{timestamp}.log`

## 🔍 Log Content Examples

### Case Log Structure
```
=== NEW CASE #1 STARTED ===
Case log file: logs/volatility_case_1_20250919_003745_tick70.log
Case tick: 70
Processing tick 70/300 (status: ACTIVE)
Portfolio Delta: -10498, Required Hedge: 10498
Found 6 trading opportunities
Top opportunity: RTM51C SELL - Price diff: $1.48, Vol diff: 4.946
Executing option trade: SELL 1 RTM51C (trade #1)
...
Performance Summary:
  - Case number: 1
  - Log file: logs/volatility_case_1_20250919_003745_tick70.log
  - Total trades executed: 22
  - Maximum delta exposure: 10498
  - Total penalties incurred: $1250.50
  - Average penalty per second: $4.17
=== CASE #1 COMPLETED ===
```

### Summary Log Structure
```
=== CASE #1 SUMMARY ===
Timestamp: 2025-09-19 00:37:45.123456
Log file: logs/volatility_case_1_20250919_003745_tick70.log
Total trades: 22
Max delta exposure: 10498
Total penalties: $1250.50
Avg penalty/second: $4.17
==================================================

=== CASE #2 SUMMARY ===
Timestamp: 2025-09-19 00:42:30.789012
Log file: logs/volatility_case_2_20250919_004230_tick0.log
Total trades: 15
Max delta exposure: 8500
Total penalties: $980.25
Avg penalty/second: $3.27
==================================================
```

## 🛠️ Log Management Tools

### 1. **Log Viewer Script** (`view_logs.py`)
A Python script to help manage and view log files:

```bash
python view_logs.py
```

**Features:**
- List all available log files
- Show case summaries
- View latest case logs
- Interactive menu system

### 2. **Manual Log Analysis**
```bash
# List all case logs
ls logs/volatility_case_*.log

# View latest case log
tail -50 logs/volatility_case_*.log | head -50

# View summary for today
cat logs/volatility_summary_$(date +%Y%m%d).log
```

## 📊 Performance Analysis

### Key Metrics Tracked
- **Total trades executed**: Number of trades per case
- **Maximum delta exposure**: Peak delta exposure reached
- **Total penalties**: Cumulative penalties incurred
- **Average penalty per second**: Penalty rate
- **Case duration**: Time from start to completion

### Analysis Examples
```bash
# Find cases with high penalties
grep "Total penalties" logs/volatility_summary_*.log | sort -k3 -nr

# Find cases with most trades
grep "Total trades" logs/volatility_summary_*.log | sort -k3 -nr

# Find cases with highest delta exposure
grep "Max delta exposure" logs/volatility_summary_*.log | sort -k4 -nr
```

## 🔧 Configuration

### Log File Settings
- **Directory**: `logs/` (created automatically)
- **Format**: `%(asctime)s - %(levelname)s - %(message)s`
- **Level**: `INFO` (includes all trading decisions)
- **Rotation**: New file per case

### Customization
To modify logging behavior, edit the `setup_logging()` function in `volatility.py`:

```python
def setup_logging(filename):
    """Setup logging with a specific filename"""
    # Modify handlers, format, or level here
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for more detail
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(filename, mode='w')
        ]
    )
```

## 📈 Benefits

### 1. **Case Isolation**
- Each case has its own log file
- Easy to compare performance across cases
- No mixing of different trading sessions

### 2. **Performance Tracking**
- Detailed metrics for each case
- Easy to identify best/worst performing cases
- Historical performance analysis

### 3. **Debugging**
- Isolate issues to specific cases
- Detailed trading decisions and outcomes
- Error tracking per case

### 4. **Analysis**
- Compare strategy performance over time
- Identify patterns in successful cases
- Optimize strategy based on historical data

## 🚀 Usage

### Running the Strategy
```bash
python volatility.py
```

### Viewing Logs
```bash
# Use the log viewer
python view_logs.py

# Or manually view files
ls logs/
cat logs/volatility_summary_20250919.log
```

### Cleaning Old Logs
```bash
# Remove logs older than 7 days
find logs/ -name "*.log" -mtime +7 -delete

# Archive logs
tar -czf logs_archive_$(date +%Y%m%d).tar.gz logs/
```

## 📝 Best Practices

1. **Regular Cleanup**: Archive or delete old log files periodically
2. **Monitor Disk Space**: Log files can grow large with detailed logging
3. **Case Analysis**: Review summary logs to identify performance patterns
4. **Backup Important Cases**: Save logs from successful trading sessions
5. **Performance Monitoring**: Track penalty trends across cases

## 🔍 Troubleshooting

### Common Issues
- **No logs directory**: Strategy will create it automatically
- **Permission errors**: Ensure write access to the logs directory
- **Large log files**: Consider reducing log level or implementing log rotation
- **Missing case logs**: Check if cases are being detected properly

### Debug Mode
To enable more detailed logging, change the log level in `setup_logging()`:
```python
level=logging.DEBUG  # More detailed logging
```

