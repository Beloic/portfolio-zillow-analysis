# 📊 Zillow Home Value Index Dataset

## Complete Dataset

The full `Zillow_Home_Value_Index.csv` dataset is too large for GitHub (>100MB).

### How to obtain the complete dataset:

1. **Direct Download**: Visit [Zillow Research](https://www.zillow.com/research/) and download the Zillow Home Value Index
2. **Alternative**: Use the provided sample `Zillow_Home_Value_Index_sample.csv` for testing analyses

## Provided Sample

- **File**: `Zillow_Home_Value_Index_sample.csv`
- **Size**: ~1000 regions (representative sample)
- **Period**: 2000-2025 (monthly data)
- **Usage**: Perfect for demonstrations and testing

## Data Structure

| Column | Description |
|--------|-------------|
| RegionID | Unique region identifier |
| SizeRank | Region size ranking |
| RegionName | Region name |
| RegionType | Type (zip, city, county, state) |
| StateName | State name |
| State | State code |
| City | City name |
| Metro | Metropolitan area |
| CountyName | County name |
| [Dates] | Average prices by month (2000-01-31 to 2025-01-31) |

## Data Quality Notes

- **Missing Values**: Some regions may have missing data for certain time periods
- **Data Types**: Price columns are numeric, geographic columns are categorical
- **Time Series**: Monthly data points spanning 25+ years
- **Geographic Coverage**: All US states and major metropolitan areas

## Important Note

For complete analyses, use the full dataset downloaded from Zillow Research. The sample provided is sufficient for portfolio demonstration and model development.
