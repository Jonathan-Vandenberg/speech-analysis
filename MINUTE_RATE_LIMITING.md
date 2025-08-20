# Per-Minute Rate Limiting Feature

This document describes the per-minute rate limiting feature added to the Speech Analysis API.

## Overview

The API now supports three levels of rate limiting:
- **Per-minute**: Limits requests within a rolling minute window
- **Daily**: Limits requests per calendar day
- **Monthly**: Limits requests per calendar month

## Database Changes

### New Columns in `api_keys` Table
- `minute_limit`: Maximum requests allowed per minute (default: 10)
- `minute_usage`: Current usage in the current minute
- `last_minute_reset`: Timestamp of last minute counter reset

### Migration for Existing Databases

If you already have an `api_keys` table, run this migration:

```sql
-- Add minute limit columns
ALTER TABLE api_keys 
ADD COLUMN IF NOT EXISTS minute_limit INTEGER DEFAULT 10,
ADD COLUMN IF NOT EXISTS minute_usage INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS last_minute_reset TIMESTAMPTZ DEFAULT DATE_TRUNC('minute', NOW());
```

Or run the complete migration file:
```bash
# Execute the migration script in Supabase
psql -f database_migration_add_minute_limits.sql
```

## Backend Changes

### Rate Limiting Logic
1. **Validation Order**: Minute → Daily → Monthly limits
2. **Counter Updates**: All three counters increment with each request
3. **Automatic Reset**: Counters reset automatically based on time windows

### Error Messages
- `Per-minute limit exceeded (X requests/minute)`
- `Daily limit exceeded (X requests/day)`  
- `Monthly limit exceeded (X requests/month)`

### Database Function
The `reset_usage_counters()` function now resets minute counters:
```sql
UPDATE api_keys 
SET minute_usage = 0, last_minute_reset = DATE_TRUNC('minute', NOW())
WHERE last_minute_reset < DATE_TRUNC('minute', NOW());
```

## Frontend Changes

### Create API Key Form
New field for setting per-minute limits:
- **Field**: "Per-Minute Limit"
- **Default**: 10 requests/minute
- **Range**: 1-1000 requests/minute

### API Keys Table
New column showing minute usage:
- **Column**: "Per-Minute"
- **Format**: `current/limit (percentage%)`
- **Color coding**: Red (>80%), Yellow (>60%), Green (≤60%)

### Dashboard Statistics
New metric card:
- **Title**: "Current Minute"
- **Value**: Total minute usage across all API keys
- **Description**: "This minute"

## API Usage

### Creating API Keys
```bash
curl -X POST "http://localhost:8000/api/admin/keys" \
  -d "description=My API Key" \
  -d "minute_limit=20" \
  -d "daily_limit=1000" \
  -d "monthly_limit=10000"
```

### Response Format
```json
{
  "api_key": "sk-...",
  "key_id": "uuid",
  "description": "My API Key",
  "minute_limit": 20,
  "daily_limit": 1000,
  "monthly_limit": 10000
}
```

## Use Cases

### High-Frequency Applications
- **Limit**: 100 requests/minute
- **Use Case**: Real-time speech analysis in live applications
- **Daily/Monthly**: Higher limits to accommodate sustained usage

### Standard Applications  
- **Limit**: 10 requests/minute (default)
- **Use Case**: Regular batch processing or user-driven analysis
- **Daily/Monthly**: Standard limits for normal usage patterns

### Rate-Limited Applications
- **Limit**: 1-5 requests/minute
- **Use Case**: Free tier or testing accounts
- **Daily/Monthly**: Lower limits to prevent abuse

## Rate Limiting Strategy

### Enforcement Order
1. **Check minute limit** (most restrictive, immediate feedback)
2. **Check daily limit** (prevents sustained abuse)
3. **Check monthly limit** (billing period protection)

### Counter Management
- **Minute counters**: Reset every minute using `DATE_TRUNC('minute', NOW())`
- **Daily counters**: Reset every day at midnight
- **Monthly counters**: Reset on the first day of each month

### Performance Considerations
- **Database calls**: Each request triggers 3 counter updates
- **Optimization**: Counters are updated in a single query
- **Caching**: Consider Redis for high-traffic scenarios

## Monitoring and Analytics

### Admin Dashboard
- **Real-time monitoring**: Current minute usage across all keys
- **Usage patterns**: Historical per-minute usage trends
- **Rate limit violations**: Tracking of rejected requests

### Database Views
The `api_usage_analytics` view includes minute-level statistics for comprehensive analysis.

## Migration Guide

### For Existing Installations

1. **Update Database Schema**:
   ```bash
   # Run migration script
   psql -f database_migration_add_minute_limits.sql
   ```

2. **Update Backend Code**:
   ```bash
   # Pull latest changes
   git pull origin main
   
   # Install any new dependencies
   pip install -r requirements.txt
   ```

3. **Update Frontend**:
   ```bash
   cd school-ai-admin
   npm install  # Install any new dependencies
   ```

4. **Restart Services**:
   ```bash
   # Restart API server
   uvicorn main.app:app --reload
   
   # Restart admin frontend
   npm run dev
   ```

### Default Values
- **New API keys**: 10 requests/minute (configurable)
- **Existing API keys**: Will be updated with default of 10 requests/minute
- **Admin key**: 100 requests/minute (higher default for admin usage)

## Testing

### Validation Script
```bash
# Test the per-minute rate limiting
python test_api_integration.py
```

### Manual Testing
1. Create an API key with a low minute limit (e.g., 2 requests/minute)
2. Make multiple rapid requests
3. Verify rate limiting kicks in after the limit is exceeded
4. Wait 1 minute and verify counter resets

## Security Considerations

### Rate Limiting Benefits
- **DDoS Protection**: Prevents rapid-fire attacks
- **Resource Protection**: Limits computational load per user
- **Fair Usage**: Ensures equitable access across all users

### Implementation Notes
- **Rolling Windows**: Minute limits use exact minute boundaries
- **Database Consistency**: All counter updates are atomic
- **Error Handling**: Graceful degradation when database is unavailable
