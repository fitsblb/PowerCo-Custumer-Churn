# Data Card: PowerCo Customer Churn

## Dataset Overview

### Client Data (Static Profile)
- **Records**: 14,606 customers (one per customer)
- **Duplicates**: None (all IDs unique)
- **Nulls**: 0 across all 26 features
- **Date Range**: Activation 2003–2014, contract end 2016–2017, price updates 2003–2016
- **Temporal Anchor**: Observation point (m_ref) ≈ 2015 (price data) to 2016–2017 (churn labels)

### Price Data (Time Series)
- **Records**: 193,002 price snapshots
- **Customers with price data**: 16,096 unique IDs
- **Avg records per customer**: ~12 (range varies, TBD)
- **Date Range**: 2015-01-01 to 2015-12-01 (12 months)
- **Pricing dimensions**: 6 variables (off-peak/peak/mid-peak, variable + fixed)

### Dataset Alignment
| Cohort | Count | Notes |
|--------|-------|-------|
| Both profile + price | 14,606 | ✅ Analysis set |
| Price only (no profile) | 1,490 | ❌ Cannot build features; exclude |
| Profile only (no price) | 0 | N/A |

**Working dataset**: 14,606 customers with complete data.

---

## Target Variable: Churn

- **Definition**: Binary (0/1) indicating whether customer's contract ended by observation cutoff
- **Class distribution**: 
  - Non-churned (0): 13,187 (90.3%)
  - Churned (1): 1,419 (9.7%)
  - **Imbalance ratio**: ~1:9.3 (moderate; SMOTE candidate)
- **Churned contract end dates**: 2016-01-28 to 2017-01-28 (13 months)
- **Active contract end dates**: 2016-01-28 to 2017-06-13 (17 months)

### Churn Window (TBD)
**Question**: What is the exact cutoff for labeling churn?
- Option A: `date_end <= 2016-01-31` (1-month window post-price-snapshot)
- Option B: `date_end <= 2016-12-31` (1-year window)
- Option C: Infer from distribution

---

## Feature Space (26 columns)

### Identifiers
- `id`: Customer company identifier (hashed text)

### Temporal
- `date_activ`: Contract activation date
- `date_end`: Contract end date (observed for all customers)
- `date_modif_prod`: Last product modification
- `date_renewal`: Next renewal date

### Consumption (Actual)
- `cons_12m`: Electricity consumption past 12 months (int)
- `cons_gas_12m`: Gas consumption past 12 months (int)
- `cons_last_month`: Electricity consumption last month (int)

### Consumption (Forecasted)
- `forecast_cons_12m`: Forecasted electricity 12 months (float)
- `forecast_cons_year`: Forecasted electricity calendar year (int)

### Pricing & Margins
- `forecast_price_energy_off_peak`: Forecasted off-peak energy price (float)
- `forecast_price_energy_peak`: Forecasted peak energy price (float)
- `forecast_price_pow_off_peak`: Forecasted off-peak power price (float)
- `forecast_discount_energy`: Forecasted discount value (float)
- `forecast_meter_rent_12m`: Forecasted meter rental bill 12m (float)
- `margin_gross_pow_ele`: Gross margin on power subscription (float)
- `margin_net_pow_ele`: Net margin on power subscription (float)
- `net_margin`: Total net margin (float)

### Contract & Engagement
- `channel_sales`: Sales channel code (hashed text)
- `has_gas`: Gas customer flag (object; likely Y/N or 0/1)
- `imp_cons`: Current paid consumption (float)
- `nb_prod_act`: Number of active products/services (int)
- `num_years_antig`: Customer antiquity in years (int)
- `origin_up`: Electricity campaign code (hashed text)
- `pow_max`: Subscribed power (float)

### Target
- `churn`: Binary churn flag (1 = churned, 0 = active)

---

## Price Data Schema (8 columns)

- `id`: Customer identifier
- `price_date`: Reference date (2015-01-01 to 2015-12-01, monthly)
- `price_off_peak_var`: Off-peak variable energy price (float)
- `price_peak_var`: Peak variable energy price (float)
- `price_mid_peak_var`: Mid-peak variable energy price (float)
- `price_off_peak_fix`: Off-peak fixed power price (float)
- `price_peak_fix`: Peak fixed power price (float)
- `price_mid_peak_fix`: Mid-peak fixed power price (float)

---

## Data Quality Status

| Criterion | Status | Notes |
|-----------|--------|-------|
| Nulls | ✅ 0 | Clean; no imputation needed |
| Duplicates | ✅ 0 | No duplicate customer IDs in client data |
| ID alignment | ✅ 100% in analysis set | 14,606 customers have both profile + price |
| Temporal ordering | 🔄 TBD | Need to verify date logic (activation < end < renewal) |
| Encoding | 🔄 TBD | Hashed strings for `id`, `channel_sales`, `origin_up` preserve meaning but obscure domain |
| Leakage risk | 🟡 MEDIUM | Prices from 2015 predict churn in 2016–2017; gap is safe, but forecast_ columns need audit |

---

## Next Steps

1. **Diagnostic Step 2.1**: Determine exact churn cutoff from `date_end` distribution
2. **Step 3**: Customer lifecycle profiling (tenure, activity type, contract structure)
3. **Step 4**: Timestamp anomaly detection (contract logic validation)
4. **Step 5**: Price data alignment and rollup strategy