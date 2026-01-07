# EquiPay Canada - Data Sources
=====================================

This document provides authoritative references for all data sources used in the EquiPay Canada project.

## Primary Data Sources

### 1. Labour Force Survey Public Use Microdata File (LFS PUMF)

**Source:** Statistics Canada  
**Catalogue Number:** 71M0001X  
**URL:** https://www150.statcan.gc.ca/n1/en/catalogue/71M0001X

The LFS PUMF contains individual-level microdata from the monthly Labour Force Survey,
the primary source of Canadian labour market statistics.

#### Historical Annual Files (2010-2023)

| Year | Download URL |
|------|--------------|
| 2010 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2010-CSV.zip |
| 2011 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2011-CSV.zip |
| 2012 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2012-CSV.zip |
| 2013 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2013-CSV.zip |
| 2014 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2014-CSV.zip |
| 2015 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2015-CSV.zip |
| 2016 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2016-CSV.zip |
| 2017 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2017-CSV.zip |
| 2018 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2018-CSV.zip |
| 2019 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2019-CSV.zip |
| 2020 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2020-CSV.zip |
| 2021 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2021-CSV.zip |
| 2022 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2022-CSV.zip |
| 2023 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2023-CSV.zip |

#### 2025 Monthly Files (January - October)

| Month | Download URL |
|-------|--------------|
| 2025-01 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/2025-01-CSV.zip |
| 2025-02 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/2025-02-CSV.zip |
| 2025-03 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/2025-03-CSV.zip |
| 2025-04 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/2025-04-CSV.zip |
| 2025-05 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/2025-05-CSV.zip |
| 2025-06 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/2025-06-CSV.zip |
| 2025-07 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/2025-07-CSV.zip |
| 2025-08 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/2025-08-CSV.zip |
| 2025-09 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/2025-09-CSV.zip |
| 2025-10 | https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/2025-10-CSV.zip |

#### Key Variables Used

| Variable | Description |
|----------|-------------|
| HRLYEARN | Hourly earnings in **dollars** (converted from cents during ETL) |
| GENDER | Gender (1=Male, 2=Female) |
| PROV | Province code |
| EDUC | Highest level of education |
| AGE_6 | Age group (6 categories) |
| NOC_10 | National Occupational Classification (10 categories) |
| NAICS_21 | North American Industry Classification System (21 categories) |
| FTPTMAIN | Full-time/Part-time status |
| UNION | Union membership status |
| FINALWT | Survey weight for population inference |

> **Note:** The raw LFS PUMF stores HRLYEARN in cents (e.g., 2500 = $25.00/hour). 
> The Parquet files store HRLYEARN in dollars after conversion during the ETL process
> (`scripts/convert_to_parquet.py`). No further conversion is needed when querying the data.

---

### 2. Geographic Boundary Files

**Source:** Statistics Canada  
**Catalogue Number:** 92-160-X  
**URL:** https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/index-eng.cfm

The 2021 Census Cartographic Boundary Files provide the geographic shapes for Canadian provinces 
and territories used in choropleth map visualizations.

| File | Description |
|------|-------------|
| lpr_000a21a_e | Provinces and Territories - Cartographic Boundary File |

---

### 3. Macroeconomic Data

The following macroeconomic indicators are used for CPI adjustment and economic context:

| Indicator | Source | Table ID |
|-----------|--------|----------|
| Consumer Price Index (CPI) | Statistics Canada | 18-10-0005-01 |
| GDP Growth | Statistics Canada | 36-10-0104-01 |
| Unemployment Rate | Statistics Canada (LFS) | 14-10-0287-01 |
| Bank of Canada Policy Rate | Bank of Canada | - |

---

## Data Download Instructions

To download the LFS PUMF data, run:

```bash
# Download all available data (2010-2023 + 2025 monthly)
python scripts/download_lfs_data.py --all

# Download specific years
python scripts/download_lfs_data.py --years 2020 2021 2022 2023

# Download specific 2025 months
python scripts/download_lfs_data.py --months 2025-01 2025-02 2025-03

# Check what data is available
python scripts/download_lfs_data.py --summary
```

---

## Data Terms of Use

Statistics Canada data is subject to the Statistics Canada Open Licence:
https://www.statcan.gc.ca/en/reference/licence

> **Statistics Canada Open Licence**
> 
> You are encouraged to use the Information that is available under this licence with only a few conditions.
> 
> Using Information under this licence:
> - Use the Information in a product or application.
> - Make copies of the Information.
> - Modify the Information.
> - Redistribute the Information.
>
> You must:
> - Acknowledge the source of the Information by including any attribution statement specified by Statistics Canada.

---

## Citation

When using this data, please cite:

> Statistics Canada. Labour Force Survey Public Use Microdata File. 
> Catalogue 71M0001X. Ottawa: Statistics Canada.

---

*Last updated: 2025-01-01*
