import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter


st.set_page_config(
    page_title="Sales Dashboard",
    page_icon=":bar_chart:",
    layout="wide")
st.title("Reckon Sales Dashboard")
st.markdown("_prototype v 0.0.1_")

@st.cache_data
def load_data(path: str):
    data = pd.read_excel(path)
    return data

with st.sidebar:
    st.header("configuration")
    uploaded_file = st.file_uploader("choose a file")

if uploaded_file is None:
    st.info(" Upload a file thorugh config ",icon="ℹ️")
    st.stop()

df = load_data(uploaded_file)
 
with st.expander("Data Preview"):
    st.dataframe(df)

@st.cache_data
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

    # Create bar chart: Revenue by Year
    fig_bar = px.bar(
        revenue_by_year,
        x="Year",
        y="Value",
        title="Revenue by Year",
        labels={"Value": "Revenue ($)", "Year": "Year"},
        text_auto=".3s"  # Short format (e.g., 100M instead of 100000000)
    )

    # Ensure years are categorical to prevent gaps
    fig_bar.update_xaxes(type="category", tickmode="linear")

    # Display the chart in Streamlit
    st.plotly_chart(fig_bar)

@st.cache_data
def display_kpis(df):
    """Calculates and displays key performance indicators (KPIs) in Streamlit."""
    
    # Ensure correct data types
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Price Per Unit"] = pd.to_numeric(df["Price Per Unit"], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["Quantity", "Value"])

    # Calculate KPIs
    total_quantity = df["Quantity"].count()
    total_value = df["Value"].sum()
    avg_price_per_unit = df["Price Per Unit"].mean()

    # Format numbers in millions (M) or thousands (K)
    total_quantity_str = f"{total_quantity/1e3:.1f}K" if total_quantity > 1e3 else f"{total_quantity:.0f}"
    total_value_str = f"₹{total_value/1e6:.1f}M" if total_value > 1e6 else f"₹{total_value:.0f}"
    avg_price_str = f"₹{avg_price_per_unit:.2f}"

    # Display KPIs
    st.subheader("📊 Key Performance Indicators")
    col1, col2, col3 = st.columns(3)
    
    col1.metric(label="Total Quantity Sold", value=total_quantity_str)
    col2.metric(label="Total Revenue", value=total_value_str)
    col3.metric(label="Avg. Price per Unit", value=avg_price_str)

@st.cache_data
def plot_pareto_chart(data):
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

@st.cache_data
def plot_revenue_by_customer_type(df):
    """Generates and displays a pie chart for Revenue by Customer Type in Streamlit."""
    
    # Aggregate sum of 'Value' by 'Customer Type'
    pie_df = df.groupby("Customer Type", as_index=False)["Value"].sum()

    # Create a pie chart
    fig_pie = px.pie(
        pie_df,
        names="Customer Type",
        values="Value",
        title="Revenue by Customer Type",
        hole=0.3,  # Creates a donut-style chart
        color_discrete_sequence=px.colors.qualitative.Pastel  # Use pastel colors
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig_pie)

@st.cache_data
def plot_pareto_chart_by_location(df):
    """
    Generates and displays a Pareto chart for Avg. Unit Price by Location (Top 10).
    - Bars (Left Y-Axis) -> Total Revenue (Sum of Value)
    - Line (Right Y-Axis) -> Average Price Per Unit
    """
    # Ensure correct data types
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Price Per Unit"] = pd.to_numeric(df["Price Per Unit"], errors="coerce")

    # Drop missing values
    df = df.dropna(subset=["Value", "Price Per Unit"])

    # Aggregate data by 'Location'
    pareto_df = df.groupby("Location", as_index=False).agg({"Value": "sum", "Price Per Unit": "mean"})

    # Sort by 'Value' in descending order
    pareto_df = pareto_df.sort_values("Value", ascending=False)

    # Keep only the top 10 locations
    pareto_df = pareto_df.head(10)

    # Formatter for bar chart values (in Millions)
    def millions(x, pos):
        return f'{x * 1e-6:.0f}M'

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Bar chart for total revenue (left y-axis)
    sns.barplot(x="Location", y="Value", data=pareto_df, color="blue", ax=ax1)
    ax1.set_ylabel("Value", color="blue")
    ax1.set_xlabel("Location")
    ax1.tick_params(axis="y", colors="blue")
    ax1.yaxis.set_major_formatter(FuncFormatter(millions))

    # Line plot for price per unit (right y-axis)
    ax2 = ax1.twinx()
    sns.lineplot(x="Location", y="Price Per Unit", data=pareto_df, color="red", marker="o", ax=ax2, label="Price Per Unit")
    ax2.set_ylabel("Price Per Unit", color="red")
    ax2.tick_params(axis="y", colors="red")

    # Title and formatting
    plt.title("Avg. Unit Price by Location (Top 10)")

    # Rotate x-axis labels for better readability
    ax1.set_xticklabels(pareto_df["Location"], rotation=45, ha="right")

    # Legends
    ax1.legend(["Value"], loc="upper left")
    ax2.legend(["Price Per Unit"], loc="upper right")

    # Display plot in Streamlit
    st.pyplot(fig)


# Call the function after loading the dataset
display_kpis(df)
col11, col12 = st.columns(2)
with col11:
    plot_revenue_by_year(df)
with col12:
    plot_revenue_by_customer_type(df)
plot_pareto_chart(df)
plot_pareto_chart_by_location(df)
