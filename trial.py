import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp 
import seaborn as sns
from PIL import Image
from matplotlib.ticker import FuncFormatter


# Set page configuration
st.set_page_config(
    page_title="Sales Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)

# Load the company logo
logo_path_1= "./reckon_logo_1.png"
logo_path_2= "./skilltelligent_1.jpeg"
client_logo = Image.open(logo_path_1)
skilltelligent_logo = Image.open(logo_path_2)

# Create columns for layout
col1, col2 = st.columns([2, 6])  

with col1:
    st.image(client_logo, width=300)  
with col2:
    st.title("**Sales Dashboard**")

st.markdown("_prototype v 0.0.1_")

# Load data function (Cached for performance)
@st.cache_data
def load_data(path: str):
    data = pd.read_excel(path)
    return data

# Sidebar Configuration
with st.sidebar:
    st.markdown("**Designed by Skilltelligent**")
    st.image(skilltelligent_logo, width=50)  # Adjust width as needed
    st.header("Configuration")
    uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is None:
    st.info("Upload a file through config", icon="ℹ️")
    st.stop()

data = load_data(uploaded_file)

# Ensure 'Date' column is in datetime format
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

# Get min and max date from the dataset
start_date = data["Date"].min()
end_date = data["Date"].max()

# Create two columns for date selection
cola, colb = st.columns(2)

with cola:
    date1 = st.date_input("Start Date", start_date)
with colb:
    date2 = st.date_input("End Date", end_date)

# Convert user-selected dates to datetime format
date1 = pd.to_datetime(date1)
date2 = pd.to_datetime(date2)

# Filter DataFrame based on the selected date range
data = data[(data["Date"] >= date1) & (data["Date"] <= date2)].copy()

# Extract Year and Month
data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.strftime("%B")  # Full month name (e.g., "January")

# Sidebar filters
st.sidebar.header("Filters")

# Reusable multi-select filter function
def multi_select_filter(label, column_name, current_df):
    """Reusable multi-select filter for Streamlit sidebar."""
    options = ["All"] + sorted(current_df[column_name].dropna().unique().tolist())
    selected = st.sidebar.multiselect(label, options, default="All")
    
    if "All" in selected or not selected:  # No filter applied
        return current_df
    else:
        return current_df[current_df[column_name].isin(selected)]

# Apply filters progressively
df = data.copy()  # Start with a full copy of data, store result in df

# Define filter order (adjust based on your desired hierarchy)
filter_order = [
    ("Select Year", "Year"),
    ("Select Month", "Month"),
    ("Select Location", "Location"),
    ("Select Dealer", "Dealer"),
    ("Select Category", "Category"),
    ("Select Product Name", "Product Name"),
    ("Select Customer Type", "Customer Type")
]

# Apply each filter in sequence
for label, column in filter_order:
    df = multi_select_filter(label, column, df)


# ********************Elements of the dashboard starts from here*******************

# KPIs metrics 
def display_kpis(df):
    """Displays key performance indicators (KPIs) in Streamlit."""
    
    # Ensure correct data types
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    
    # Calculate KPIs
    total_revenue = df["Value"].sum()
    total_orders = df["Quantity"].count()
    avg_order_value = df["Price Per Unit"].mean()


    # Format numbers for better readability
    total_revenue_str = f"₹{total_revenue/1e6:.1f}M" if total_revenue > 1e6 else f"₹{total_revenue:.0f}"
    total_orders_str = f"{total_orders/1e3:.1f}K" if total_orders > 1e3 else f"{total_orders:.0f}"
    avg_order_value_str = f"₹{avg_order_value:.2f}"

    # Display KPIs in columns
    st.subheader("🎯 Key Performance Indicators")
    col1, col2, col3 = st.columns(3)

    col1.metric(label="Total Revenue", value=total_revenue_str)
    col2.metric(label="Total Orders", value=total_orders_str)
    col3.metric(label="Avg. Order Value", value=avg_order_value_str)

# Yearly Revenue Bar Chart
def plot_revenue_by_year(df):
    """Generates and displays a properly formatted bar chart for Revenue by Year."""
    
    # Convert Year column to numeric, forcing errors to NaN
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"])  # Remove invalid years
    df["Year"] = df["Year"].astype(int)  # Convert back to integer

    # Filter out unrealistic years (e.g., before 2000)
    df = df[df["Year"] >= 2000]  # Adjust this range as needed

    # Group data by Year and sum Revenue
    revenue_by_year = df.groupby("Year", as_index=False)["Value"].sum()
    # matplotlib default blue color
    custom_blue = "#0000FF"
    # Create bar chart: Revenue by Year
    fig_bar = px.bar(
        revenue_by_year,
        x="Year",
        y="Value",
        title="Revenue by Year",
        labels={"Value": "Revenue ($)", "Year": "Year"},
        text_auto=".3s",  # Short format (e.g., 100M instead of 100000000)
        color_discrete_sequence=[custom_blue]
    )

    # Ensure years are categorical to prevent gaps
    fig_bar.update_xaxes(type="category", tickmode="linear")

    # Display the chart in Streamlit
    st.plotly_chart(fig_bar)

# Revenue by Customer Type (Pie Chart)
def plot_revenue_by_customer_type(df):
    pie_df = df.groupby("Customer Type", as_index=False)["Value"].sum()
    blue_shades = ["#0000FF", "#ADD8E6", "#87CEFA"]
    fig_pie = px.pie(
        pie_df, names="Customer Type", values="Value",
        title="Revenue by Customer Type", hole=0.3,
        color_discrete_sequence=blue_shades
    )
    st.plotly_chart(fig_pie)

# Pareto Chart: sales by Location (Top 10)
def plot_pareto_chart_by_location(df):
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Price Per Unit"] = pd.to_numeric(df["Price Per Unit"], errors="coerce")
    df = df.dropna(subset=["Value", "Price Per Unit"])

    pareto_df = df.groupby("Location", as_index=False).agg({"Value": "sum", "Price Per Unit": "mean"})
    pareto_df = pareto_df.sort_values("Value", ascending=False).head(10)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    sns.barplot(x="Location", y="Value", data=pareto_df, color="blue", ax=ax1)
    ax1.set_ylabel("Total Revenue", color="blue")
    ax1.tick_params(axis="y", colors="blue")

    ax2 = ax1.twinx()
    sns.lineplot(x="Location", y="Price Per Unit", data=pareto_df, color="red", marker="o", ax=ax2)
    ax2.set_ylabel("Avg. Unit Price", color="red")
    ax2.tick_params(axis="y", colors="red")

    plt.title("Avg. Unit Price by Location (Top 10)")
    ax1.set_xticklabels(pareto_df["Location"], rotation=45, ha="right")

    ax1.legend(["Total Revenue"], loc="upper left")
    ax2.legend(["Avg. Unit Price"], loc="upper right")

    st.pyplot(fig)

# Pareto Chart: sales by month
def plot_pareto_chart_monthly_revenue(data):
    """
    Generates and displays a Pareto chart for Monthly Revenue Analysis in Streamlit.
    Supports both file paths and already-loaded DataFrames.
    """
    # Load the dataset if a file path is provided
    if isinstance(data, str):
        df = pd.read_excel(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("Invalid input: Provide a valid file path or a Pandas DataFrame.")

    # Convert 'Date' to datetime and extract month names
    df["Month"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%B")

    # Aggregate sum of 'Value' by Month
    pareto_df = df.groupby("Month", observed=True)["Value"].sum().reset_index()

    # Sort by descending Value (Pareto Rule)
    pareto_df = pareto_df.sort_values("Value", ascending=False)

    # Calculate cumulative percentage
    pareto_df["Cumulative%"] = pareto_df["Value"].cumsum() / pareto_df["Value"].sum() * 100

    # Calculate contribution percentage (Contri%)
    pareto_df["Contri%"] = (pareto_df["Value"] / pareto_df["Value"].sum()) * 100

    # Formatter to show values in millions (M)
    def millions(x, pos):
        return f'{x * 1e-6:.0f}M'

    # Plot Pareto chart
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Bar plot (Sum of Value) - Sorted by Value
    sns.barplot(x="Month", y="Value", data=pareto_df, color="blue", ax=ax1)
    ax1.set_ylabel("Value", color="blue")
    ax1.set_xlabel("Month")
    ax1.tick_params(axis="y", colors="blue")
    ax1.yaxis.set_major_formatter(FuncFormatter(millions))

    # Line plot (Cumulative %)
    ax2 = ax1.twinx()
    sns.lineplot(x="Month", y="Cumulative%", data=pareto_df, color="red", marker="o", ax=ax2, label="Cumulative%")
    ax2.set_ylabel("Cumulative %", color="red")
    ax2.tick_params(axis="y", colors="red")
    ax2.set_ylim(0, 120)

    # Line plot (Contri %)
    sns.lineplot(x="Month", y="Contri%", data=pareto_df, color="purple", marker="o", ax=ax2, label="Contri%")

    # Title and formatting
    plt.title("Revenue by Month - Pareto Chart")

    # Tilt month names diagonally for better readability
    ax1.set_xticklabels(pareto_df["Month"], rotation=45, ha="right")

    ax1.legend(["Value"], loc="upper left")
    ax2.legend(loc="upper right")

    # Display plot in Streamlit
    st.pyplot(fig)

# Pareto Chart: sales by Dealer (Top 10)
def plot_pareto_chart_by_dealer(df):
    """
    Generates a Pareto chart for revenue and average unit price by Dealer.
    """

    # Convert relevant columns to numeric
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Price Per Unit"] = pd.to_numeric(df["Price Per Unit"], errors="coerce")

    # Drop rows with missing values in 'Value' or 'Price Per Unit'
    df = df.dropna(subset=["Value", "Price Per Unit"])

    # Aggregate total revenue and average price per unit by Dealer
    pareto_df = df.groupby("Dealer", as_index=False).agg({"Value": "sum", "Price Per Unit": "mean"})

    # Sort by total revenue in descending order and select top 10 dealers
    pareto_df = pareto_df.sort_values("Value", ascending=False).head(10)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Bar plot for total revenue per dealer
    sns.barplot(x="Dealer", y="Value", data=pareto_df, color="blue", ax=ax1)
    ax1.set_ylabel("Total Revenue", color="blue")
    ax1.tick_params(axis="y", colors="blue")

    # Secondary axis for average unit price
    ax2 = ax1.twinx()
    sns.lineplot(x="Dealer", y="Price Per Unit", data=pareto_df, color="red", marker="o", ax=ax2)
    ax2.set_ylabel("Avg. Unit Price", color="red")
    ax2.tick_params(axis="y", colors="red")

    # Title and formatting
    plt.title("Avg. Unit Price by Dealer (Top 10)")
    ax1.set_xticklabels(pareto_df["Dealer"], rotation=45, ha="right")

    ax1.legend(["Total Revenue"], loc="upper left")
    ax2.legend(["Avg. Unit Price"], loc="upper right")

    # Display the plot in Streamlit
    st.pyplot(fig)

# Pareto Chart: sales by product
def plot_pareto_chart_by_product(data):
    """
    Generates and displays a Pareto chart for the Top 10 Products by Revenue in Streamlit.
    Supports both file paths and already-loaded DataFrames.
    """
    try:
        # Load dataset if given as a file path
        if isinstance(data, str):
            df = pd.read_excel(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()  # Avoid modifying the original DataFrame
        else:
            raise ValueError("Invalid input: Provide a valid file path or a Pandas DataFrame.")

        # Aggregate revenue by Product Name
        pareto_df = df.groupby("Product Name", observed=True)["Value"].sum().reset_index()

        # Sort by descending revenue (Pareto Rule) & take Top 10
        pareto_df = pareto_df.sort_values("Value", ascending=False).head(10)

        # Calculate cumulative percentage
        pareto_df["Cumulative%"] = pareto_df["Value"].cumsum() / pareto_df["Value"].sum() * 100

        # Calculate contribution percentage (Contri%)
        pareto_df["Contri%"] = (pareto_df["Value"] / pareto_df["Value"].sum()) * 100

        # Formatter for millions
        def millions(x, pos):
            return f'{x * 1e-6:.1f}M'

        # Plot setup
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Bar plot (Total Revenue)
        sns.barplot(x="Product Name", y="Value", data=pareto_df, color="blue", ax=ax1)
        ax1.set_ylabel("Total Revenue", color="blue")
        ax1.set_xlabel("Product Name")
        ax1.tick_params(axis="y", colors="blue")
        ax1.yaxis.set_major_formatter(FuncFormatter(millions))

        # Line plot (Cumulative % & Contribution %)
        ax2 = ax1.twinx()
        sns.lineplot(x="Product Name", y="Cumulative%", data=pareto_df, color="red", marker="o", ax=ax2, label="Cumulative%")
        sns.lineplot(x="Product Name", y="Contri%", data=pareto_df, color="purple", marker="o", ax=ax2, label="Contri%")
        ax2.set_ylabel("Percentage (%)", color="red")
        ax2.tick_params(axis="y", colors="red")
        ax2.set_ylim(0, 120)

        # Title
        plt.title("Top 10 Products by Revenue - Pareto Chart")

        # Improve readability of x-axis (rotate product names)
        ax1.set_xticklabels(pareto_df["Product Name"], rotation=45, ha="right")

        # Legends
        ax1.legend(["Total Revenue"], loc="upper left")
        ax2.legend(loc="upper right")

        # Display in Streamlit
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error generating Pareto chart: {e}")

# Pivot tables
def create_pivot_table(df):
    """
    Creates and displays two pivot tables:
    1. Value as Sum (Percentage of Total)
    2. Dealer as Distinct Count
    """

    # Ensure required columns exist in the dataset
    required_columns = ["Date", "Registration Month", "Registation Month & Year", "Value", "Dealer"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"Missing columns in dataset: {', '.join(missing_columns)}")
        return

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Pivot table for Value (sum as percentage of total)
    value_pivot = df.pivot_table(
        values="Value",
        index="Registration Month",
        columns="Registation Month & Year",
        aggfunc="sum",
        fill_value=0
    )

    # Convert sum to percentage of total
    value_pivot_percentage = value_pivot.div(value_pivot.sum().sum()) * 100

    # Pivot table for Dealer (distinct count)
    dealer_pivot = df.pivot_table(
        values="Dealer",
        index="Registration Month",
        columns="Registation Month & Year",
        aggfunc=pd.Series.nunique,  # Unique count of dealers
        fill_value=0
    )

    # Display the pivot table
    col41,col42 = st.columns(2)
    with col41:
        st.markdown("<h4 style='text-align: left; font-size:18px;'>Monthly Sales Distribution (% of Total Sales)</h4>", unsafe_allow_html=True)
        st.dataframe(value_pivot_percentage)
    with col42:
        st.markdown("<h4 style='text-align: left; font-size:18px;'>New Dealer Onboarding Trend (Unique Count)</h4>", unsafe_allow_html=True)
        st.dataframe(dealer_pivot)

# function calling 
display_kpis(df)
st.subheader("📊 Visualizations")
col11, col12 = st.columns(2)
col21, col22 = st.columns(2)
col31, col32 = st.columns(2)
create_pivot_table(df)

with col11:
    plot_revenue_by_year(df)
with col12:
    plot_revenue_by_customer_type(df)
with col21:
    plot_pareto_chart_by_location(df)
with col22:
    plot_pareto_chart_monthly_revenue(df)
with col31:
    plot_pareto_chart_by_dealer(df)
with col32:
    plot_pareto_chart_by_product(df)

