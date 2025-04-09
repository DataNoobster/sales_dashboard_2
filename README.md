# sales_dashboard_app

## Overview

The **Sales Dashboard** is a dynamic, interactive web application built with [Streamlit](https://streamlit.io/) to empower businesses with deep insights into their sales data. Crafted by **Skilltelligent**, this prototype (v0.0.2) transforms raw Excel data into actionable intelligence through intuitive KPIs, rich visualizations, and AI-powered summaries via the Google Gemini API. Whether you're tracking revenue trends, analyzing customer behavior, or optimizing inventory, this tool delivers a seamless experience for data-driven decision-making.

With support for flexible filtering, exportable datasets, and visually stunning [Plotly](https://plotly.com/) charts, the Sales Dashboard is designed to help teams uncover patterns and opportunities effortlessly.

## Features

- **Data Upload**: Seamlessly upload sales data in Excel (.xlsx) format for instant analysis.
- **Dynamic Filters**: Slice and dice data by year, month, location, dealer, category, product name, and customer type, with an intuitive "All" option.
- **Key Performance Indicators (KPIs)**:
  - Total Revenue (₹, smartly formatted in millions or thousands)
  - Total Orders (scaled to thousands where applicable)
  - Average Order Value
- **Visualizations**:
  - Bar Chart: Revenue by Year
  - Pie Chart: Revenue by Customer Type
  - Pareto Charts: Top 10 Revenue by Location, Month, Dealer, and Product
- **Pivot Tables**:
  - Monthly Sales Distribution (% of total sales)
  - New Dealer Onboarding Trend (unique dealer count)
- **Data Export**: Download filtered datasets as Excel files and pivot tables as CSVs with a single click.
- **AI-Powered Summary**: Generate concise executive reports with insights and recommendations, powered by the Google Gemini API.
- **Responsive Layout**: A wide, user-friendly interface with expandable sections for data previews and summaries.

## Tech Stack

- **Python 3.8+**
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Pillow (PIL)**: Image processing for logos
- **Google Gemini API**: AI-driven summary generation
- **NumPy**: Numerical computations

## Prerequisites

To run or contribute to this project, ensure you have:

- **Python Version**: 3.8 or higher
- **LLM API Key**: A Google Gemini API key (required for summary generation), though you can swap it with other LLMs like Claude, OpenAI, etc., with minor code adjustments

## Usage

1. **Launch the Dashboard**:
   - Run the script with: `streamlit run main.py` 

2. **Upload Your Data**:
   - Use the sidebar to upload an Excel file. Expected columns include:
     - `Date`
     - `Value` (sales revenue)
     - `Quantity`
     - `Price Per Unit`
     - `Location`
     - `Dealer`
     - `Category`
     - `Product Name`
     - `Customer Type`
   - Optional columns for pivot tables: `Registration Month`, `Registation Month & Year`.

3. **Explore Insights**:
   - Set date ranges and apply filters to focus on specific data segments.
   - Dive into KPIs, charts, and pivot tables on the main panel.
   - Peek at the filtered dataset in the "Data Preview" expander.
   - Generate an AI-crafted summary under the "Summary Report" section.

4. **Export Results**:
   - Save your filtered data or pivot tables directly from the interface.
