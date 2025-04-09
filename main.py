import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io
import requests
import os
import google.generativeai as genai
import numpy as np


# Set page configuration
st.set_page_config(
    page_title="Sales Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode
st.markdown("""
    <style>
    .main { 
        padding: 10px; 
        background-color: #1e1e1e;  /* Dark gray background */
    }
    .stSidebar { 
        background-color: #2d2d2d;  /* Slightly lighter dark gray for sidebar */
    }
    .stButton>button { 
        background-color: #1e90ff;  /* Lighter blue for buttons */
        color: #ffffff; 
        border-radius: 5px; 
    }
    .stMetric { 
        background-color: #3a3a3a;  /* Medium gray for metrics */
        border-radius: 10px; 
        padding: 10px; 
        color: #e0e0e0;  /* Light text for readability */
    }
    h1, h2, h3, h4 { 
        color: #e0e0e0;  /* Light gray text for headers */
    }
    .stDataFrame { 
        border: 1px solid #555555;  /* Medium gray border */
        border-radius: 5px; 
        background-color: #1e1e1e; 
        color: #e0e0e0; 
    }
    div[data-testid="stExpander"] { 
        background-color: #1e1e1e; 
        color: #e0e0e0; 
    }
    </style>
""", unsafe_allow_html=True)

# Load the company logos
logo_path_1 = "./reckon_logo_1.png"
logo_path_2 = "./skilltelligent_1.jpeg"
client_logo = Image.open(logo_path_1)
skilltelligent_logo = Image.open(logo_path_2)

# Header layout with logos and title
col1, col2 = st.columns([2, 5])
with col1:
    st.image(client_logo, width=350)
with col2:
    st.title("Sales Dashboard")
st.markdown("<p style='font-size: 14px; color: #e0e0e0;'>_Prototype v0.0.2 | Powered by Skilltelligent_</p>", unsafe_allow_html=True)

# Load data function (Cached for performance)
@st.cache_data
def load_data(path):
    try:
        data = pd.read_excel(path)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Sidebar Configuration
with st.sidebar:
    st.image(skilltelligent_logo, width=100)
    st.markdown("<strong style='color: #e0e0e0;'>Designed by Skilltelligent</strong>", unsafe_allow_html=True)
    st.header("Configuration")
    
    uploaded_file = st.file_uploader("Upload Sales Data (Excel)", type=["xlsx"], help="Upload an Excel file to analyze sales data")
    
    if uploaded_file is None:
        st.info("Please upload a file to proceed", icon="ℹ️")
        st.stop()
    
    data = load_data(uploaded_file)
    if data is None:
        st.stop()

    st.subheader("Filters")

    def update_filter(key):
        """Callback function to dynamically update session state"""
        selected = st.session_state[key]
        
        if "All" in selected and len(selected) > 1:
            selected.remove("All")  # Remove "All"
        elif not selected:
            selected.append("All")  # Reset to "All" if empty

        st.session_state[key] = selected  # Update session state


    def multi_select_filter(label, column_name, current_df):
        options = sorted(current_df[column_name].dropna().unique().tolist())
        key = f"filter_{column_name}"  # Unique key for session state

        # Initialize session state for the filter if not already set
        if key not in st.session_state:
            st.session_state[key] = ["All"]

        # Create multi-select with a callback to handle "All" logic
        selected = st.multiselect(
            label, 
            ["All"] + options, 
            default=st.session_state[key], 
            key=key, 
            help=f"Filter by {label.lower()}",
            on_change=update_filter, 
            args=(key,)  # Pass the key to the callback function
        )

        # Filter the dataframe based on selection
        return current_df if "All" in selected else current_df[current_df[column_name].isin(selected)]


    # Define filter order
    filter_order = [
        ("Select Year", "FY"),
        ("Select Month", "Month"),
        ("Select Location", "Location"),
        ("Select Dealer", "Dealer"),
        ("Select Category", "Category"),
        ("Select Product Name", "Product Name"),
        ("Select Customer Type", "Customer Type-2")
        ("Dealer Or Not", "Dealer non Dealer")
    ]

    # Apply filters
    df = data.copy()
    for label, column in filter_order:
        df = multi_select_filter(label, column, df)

# Dashboard Content
st.subheader("🎯 Key Performance Indicators")
def display_kpis(df):
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    total_revenue = df["Value"].sum()
    total_orders = df["Quantity"].count()
    avg_order_value = df["Price Per Unit"].mean() if "Price Per Unit" in df.columns else 0

    total_revenue_str = f"₹{total_revenue/1e6:.1f}M" if total_revenue > 1e6 else f"₹{total_revenue:.0f}"
    total_orders_str = f"{total_orders/1e3:.1f}K" if total_orders > 1e3 else f"{total_orders:.0f}"
    avg_order_value_str = f"₹{avg_order_value:.2f}" if avg_order_value > 0 else "N/A"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Revenue", total_revenue_str, help="Total sales revenue in the selected period")
    with col2:
        st.metric("Total Orders", total_orders_str, help="Number of orders placed")
    with col3:
        st.metric("Avg. Order Value", avg_order_value_str, help="Average price per unit sold")

display_kpis(df)

# Visualizations Section
st.subheader("📊 Visualizations")
tab1, tab2 = st.tabs(["Charts", "Pivot Tables"])

with tab1:
    col11, col12 = st.columns(2)
    col21, col22 = st.columns(2)
    col31, col32 = st.columns(2)

    def plot_revenue_by_year(df):
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").dropna().astype(int)
        df = df[df["Year"] >= 2000]
        revenue_by_year = df.groupby("Year", as_index=False)["Value"].sum()
        fig = px.bar(
            revenue_by_year, x="Year", y="Value", title="Revenue by Year",
            labels={"Value": "Revenue (₹)", "Year": "Year"}, text_auto=".3s",
            color_discrete_sequence=["#1e90ff"]  # Lighter blue for dark mode
        )
        fig.update_xaxes(type="category", tickfont=dict(color="#e0e0e0"))
        fig.update_yaxes(tickfont=dict(color="#e0e0e0"))
        fig.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0")
        )
        st.plotly_chart(fig, use_container_width=True)

    def plot_revenue_by_customer_type(df):
        pie_df = df.groupby("Customer Type-2", as_index=False)["Value"].sum()
        fig = px.pie(
            pie_df, names="Customer Type-2", values="Value", title="Revenue by Customer Type",
            hole=0.3, color_discrete_sequence=["#0000FF", "#4682B4", "#00CED1"]  # Updated colors
        )
        fig.update_traces(textinfo="percent+label", textfont=dict(color="#e0e0e0"))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            legend=dict(font=dict(color="#e0e0e0"))
        )
        st.plotly_chart(fig, use_container_width=True)

    def plot_pareto_chart_by_location(df):
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df["Price Per Unit"] = pd.to_numeric(df["Price Per Unit"], errors="coerce")
        df = df.dropna(subset=["Value", "Price Per Unit"])
        pareto_df = df.groupby("Location", as_index=False).agg({"Value": "sum", "Price Per Unit": "mean"})
        pareto_df = pareto_df.sort_values("Value", ascending=False).head(10)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=pareto_df["Location"], y=pareto_df["Value"], name="Total Revenue", marker_color="#1e90ff"), secondary_y=False)
        fig.add_trace(go.Scatter(x=pareto_df["Location"], y=pareto_df["Price Per Unit"], name="Avg. Unit Price", mode="lines+markers", line=dict(color="#ff6347"), marker=dict(size=8)), secondary_y=True)
        fig.update_layout(
            title="Avg. Unit Price by Location (Top 10)", 
            xaxis=dict(title="Location", tickangle=45, tickfont=dict(color="#e0e0e0")),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.2)", bordercolor="#555555", borderwidth=1, font=dict(color="#e0e0e0")),
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0")
        )
        fig.update_yaxes(title_text="Total Revenue (₹)", title_font_color="#1e90ff", tickfont_color="#1e90ff", secondary_y=False)
        fig.update_yaxes(title_text="Avg. Unit Price (₹)", title_font_color="#ff6347", tickfont_color="#ff6347", secondary_y=True, showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    def plot_pareto_chart_monthly_revenue(df):
        df["Month"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%B")
        pareto_df = df.groupby("Month", observed=True)["Value"].sum().reset_index()
        pareto_df = pareto_df.sort_values("Value", ascending=False)
        pareto_df["Cumulative%"] = pareto_df["Value"].cumsum() / pareto_df["Value"].sum() * 100
        pareto_df["Contri%"] = (pareto_df["Value"] / pareto_df["Value"].sum()) * 100

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=pareto_df["Month"], y=pareto_df["Value"], name="Total Revenue", marker_color="#1e90ff", hovertemplate="%{x}: %{y:.2f}M<extra></extra>"), secondary_y=False)
        fig.add_trace(go.Scatter(x=pareto_df["Month"], y=pareto_df["Cumulative%"], name="Cumulative%", mode="lines+markers", line=dict(color="#ff6347"), marker=dict(size=8), hovertemplate="%{x}: %{y:.1f}%<extra></extra>"), secondary_y=True)
        fig.add_trace(go.Scatter(x=pareto_df["Month"], y=pareto_df["Contri%"], name="Contri%", mode="lines+markers", line=dict(color="#ba55d3"), marker=dict(size=8), hovertemplate="%{x}: %{y:.1f}%<extra></extra>"), secondary_y=True)
        fig.update_layout(
            title="Revenue by Month - Pareto Chart", 
            xaxis=dict(title="Month", tickangle=45, tickfont=dict(color="#e0e0e0")),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.2)", bordercolor="#555555", borderwidth=1, font=dict(color="#e0e0e0")),
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0")
        )
        fig.update_yaxes(title_text="Total Revenue (Millions)", title_font_color="#1e90ff", tickfont_color="#1e90ff", tickformat=".1f", secondary_y=False, rangemode="tozero")
        fig.update_yaxes(title_text="Percentage (%)", title_font_color="#ff6347", tickfont_color="#ff6347", range=[0, 120], secondary_y=True, showgrid=False)
        fig.update_traces(y=[x / 1e6 for x in pareto_df["Value"]], selector=dict(type="bar"))
        st.plotly_chart(fig, use_container_width=True)

    def plot_pareto_chart_by_dealer(df):
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df["Price Per Unit"] = pd.to_numeric(df["Price Per Unit"], errors="coerce")
        df = df.dropna(subset=["Value", "Price Per Unit"])
        pareto_df = df.groupby("Dealer", as_index=False).agg({"Value": "sum", "Price Per Unit": "mean"})
        pareto_df = pareto_df.sort_values("Value", ascending=False).head(10)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=pareto_df["Dealer"],
                             y=pareto_df["Value"],
                             name="Total Revenue",
                             marker_color="#1e90ff",
                             hovertemplate="%{x}: %{y:.2f}M<extra></extra>"),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=pareto_df["Dealer"],
                                 y=pareto_df["Price Per Unit"],
                                 name="Avg. Unit Price",
                                 mode="lines+markers",
                                 line=dict(color="#ff6347"),
                                 marker=dict(size=8),
                                 hovertemplate="%{x}: %{y:.2f}K<extra></extra>"),
                      secondary_y=True)
        fig.update_layout(
            title="Avg. Unit Price by Dealer (Top 10)",
            xaxis=dict(title="Dealer",
                       tickangle=45,
                       tickfont=dict(color="#e0e0e0")),
            legend=dict(x=0.01,
                        y=0.99,
                        bgcolor="rgba(255,255,255,0.2)",
                        bordercolor="#555555",
                        borderwidth=1,
                        font=dict(color="#e0e0e0")),
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0")
        )
        fig.update_yaxes(title_text="Total Revenue (Millions ₹)",
                         title_font_color="#1e90ff",
                         tickfont_color="#1e90ff",
                         tickformat=".2f",
                         secondary_y=False)
        fig.update_yaxes(title_text="Avg. Unit Price (Thousands ₹)",
                         title_font_color="#ff6347",
                         tickfont_color="#ff6347",
                         tickformat=".2f",
                         secondary_y=True,
                         showgrid=False)
        fig.update_traces(y=[x / 1e6 for x in pareto_df["Value"]],
                          selector=dict(type="bar"))
        fig.update_traces(y=[x / 1e3 for x in pareto_df["Price Per Unit"]],
                          selector=dict(type="scatter"))
        st.plotly_chart(fig, use_container_width=True)

    def plot_pareto_chart_by_product(df):
        pareto_df = df.groupby("Product Name", observed=True)["Value"].sum().reset_index()
        pareto_df = pareto_df.sort_values("Value", ascending=False).head(10)
        pareto_df["Cumulative%"] = pareto_df["Value"].cumsum() / pareto_df["Value"].sum() * 100
        pareto_df["Contri%"] = (pareto_df["Value"] / pareto_df["Value"].sum()) * 100

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=pareto_df["Product Name"], y=pareto_df["Value"], name="Total Revenue", marker_color="#1e90ff", hovertemplate="%{x}: %{y:.2f}M<extra></extra>"), secondary_y=False)
        fig.add_trace(go.Scatter(x=pareto_df["Product Name"], y=pareto_df["Cumulative%"], name="Cumulative%", mode="lines+markers", line=dict(color="#ff6347"), marker=dict(size=8), hovertemplate="%{x}: %{y:.1f}%<extra></extra>"), secondary_y=True)
        fig.add_trace(go.Scatter(x=pareto_df["Product Name"], y=pareto_df["Contri%"], name="Contri%", mode="lines+markers", line=dict(color="#ba55d3"), marker=dict(size=8), hovertemplate="%{x}: %{y:.1f}%<extra></extra>"), secondary_y=True)
        fig.update_layout(
            title="Top 10 Products by Revenue - Pareto Chart", 
            xaxis=dict(title="Product Name", tickangle=45, tickfont=dict(color="#e0e0e0")),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.2)", bordercolor="#555555", borderwidth=1, font=dict(color="#e0e0e0")),
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0")
        )
        fig.update_yaxes(title_text="Total Revenue (Millions)", title_font_color="#1e90ff", tickfont_color="#1e90ff", tickformat=".1f", secondary_y=False, rangemode="tozero")
        fig.update_yaxes(title_text="Percentage (%)", title_font_color="#ff6347", tickfont_color="#ff6347", range=[0, 120], secondary_y=True, showgrid=False)
        fig.update_traces(y=[x / 1e6 for x in pareto_df["Value"]], selector=dict(type="bar"))
        st.plotly_chart(fig, use_container_width=True)

    def plot_dealer_map(df):

        # Aggregate data by 'Updated Location'
        agg_df = df.groupby('Updated Location').agg({
            'Value': 'sum',           # Total revenue per location
            'Dealer': 'nunique',      # Number of unique dealers per location
            'Latitude': 'first',      # Keep first latitude
            'Longitude': 'first'      # Keep first longitude
        }).reset_index()

        agg_df = agg_df.rename(columns={
            'Value': 'Total Revenue',
            'Dealer': 'Number of Dealers'
        })

        # Remove rows with missing coordinates
        agg_df = agg_df.dropna(subset=['Latitude', 'Longitude'])

        # Create the map
        fig = px.scatter_mapbox(
            agg_df,
            lat="Latitude",
            lon="Longitude",
            hover_name="Updated Location",    # Show location name on hover
            hover_data={
                "Total Revenue": ":.2f",     # Show revenue formatted to 2 decimals
                "Number of Dealers": True    # Show number of dealers
            },
            size="Total Revenue",            # Size bubbles based on Total Revenue
            size_max=50,                     # Maximum bubble size (adjust as needed)
            zoom=4,
            center={"lat": 20.5937, "lon": 78.9629},  # Center on India
            mapbox_style="open-street-map",
            title="Dealer Revenue and Count by Location"
        )

        # Update layout for dark mode compatibility with Streamlit
        fig.update_layout(
            margin={"r":0, "t":50, "l":0, "b":0},
            height=600,  # Adjusted height to fit with other charts
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            title_font_color="#e0e0e0"
        )

        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)

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
    #plotting the map chart
    plot_dealer_map(df)

with tab2:
    def create_pivot_table(df):
        if "Registration Month" not in df.columns or "Registation Month & Year" not in df.columns or "Dealer" not in df.columns:
            st.error("Required columns for pivot tables are missing")
            return

        value_pivot = df.pivot_table(values="Value", index="Registration Month", columns="Registation Month & Year", aggfunc="sum", fill_value=0)
        value_pivot_percentage = value_pivot.div(value_pivot.sum().sum()) * 100
        dealer_pivot = df.pivot_table(values="Dealer", index="Registration Month", columns="Registation Month & Year", aggfunc=pd.Series.nunique, fill_value=0)

        col41, col42 = st.columns(2)
        with col41:
            st.markdown("<strong style='color: #e0e0e0;'>Monthly Sales Distribution (% of Total Sales)</strong>", unsafe_allow_html=True)
            st.dataframe(value_pivot_percentage.style.format("{:.2f}%"), use_container_width=True)
            st.download_button("Download Sales Data", value_pivot_percentage.to_csv(), "sales_distribution.csv", "text/csv")
        with col42:
            st.markdown("<strong style='color: #e0e0e0;'>New Dealer Onboarding Trend (Unique Count)</strong>", unsafe_allow_html=True)
            st.dataframe(dealer_pivot, use_container_width=True)
            st.download_button("Download Dealer Data", dealer_pivot.to_csv(), "dealer_trend.csv", "text/csv")

    create_pivot_table(df)

st.markdown("---")
# Dataset preview
with st.expander("Data Preview"):
    st.dataframe(df)

# Footer with export option
st.markdown("<strong style='color: #e0e0e0;'>Export Full Dataset</strong>", unsafe_allow_html=True)
buffer = io.BytesIO()
df.to_excel(buffer, index=False)
st.download_button("Download Filtered Data", buffer.getvalue(), "filtered_sales_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
with st.expander("📝 Summary Report", expanded=False):
    st.subheader("Generated Summary")

    def generate_summary(df, api_key, model="gemini-2.0-flash"):
        try:
            # Validate DataFrame
            if df.empty:
                return "DataFrame is empty. Please upload valid data."

            required_columns = ["Value", "Date", "Quantity"]
            for col in required_columns:
                if col not in df.columns:
                    return f"Missing column: {col}"

            # Convert data types safely
            df["Value"] = pd.to_numeric(df["Value"], errors="coerce").fillna(0)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            total_revenue = df["Value"].sum()
            total_orders = df.shape[0]
            avg_order_value = df["Value"].mean()
            top_location = df.groupby("Location")["Value"].sum().idxmax() if "Location" in df.columns else "N/A"
            top_dealer = df.groupby("Dealer")["Value"].sum().idxmax() if "Dealer" in df.columns else "N/A"
            top_product = df.groupby("Product Name")["Value"].sum().idxmax() if "Product Name" in df.columns else "N/A"
            date_range = f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}" if not df["Date"].isna().all() else "N/A"

            # Additional Insights Based on Filters
            df["Year"] = df["Date"].dt.year
            df["Month"] = df["Date"].dt.month_name()

            top_year = df.groupby("Year")["Value"].sum().idxmax() if "Year" in df.columns else "N/A"
            top_month = df.groupby("Month")["Value"].sum().idxmax() if "Month" in df.columns else "N/A"
            top_category = df.groupby("Category")["Value"].sum().idxmax() if "Category" in df.columns else "N/A"
            top_channel = df.groupby("Channel")["Value"].sum().idxmax() if "Channel" in df.columns else "N/A"
            top_price_range = df.groupby("Price Range")["Value"].sum().idxmax() if "Price Range" in df.columns else "N/A"
            
            # Prepare Gemini prompt with enhanced details
            prompt = f"""
            Generate a comprehensive executive summary for a sales dashboard, incorporating key insights from the filtered dataset:
            
            **Overview:**
            - Date Range: {date_range}
            - Total Revenue: ₹{total_revenue:,.0f}
            - Total Orders: {total_orders:,}
            
            **Key Performance Insights:**
            - Highest Revenue Location: {top_location} (Leading in sales performance)
            - Top Dealer: {top_dealer} (Most significant revenue contributor)
            - Best-Selling Product: {top_product} (Highest revenue generator)
            - Peak Sales Year: {top_year} (Highest revenue year)
            - Peak Sales Month: {top_month} (Best performing month)
            - Top Category: {top_category} (Most sold product category)
            - Dominant Sales Channel: {top_channel} (Most effective sales channel)
            - Price Range Impact: {top_price_range} (Most revenue-generating price segment)
            
            **Recommendations:**
            - Optimize sales strategies by focusing on peak months and years.
            - Align marketing efforts with high-performing categories and price segments.
            - Improve inventory management based on demand fluctuations.
            
            **Instructions:**
            - Strictly base insights on the filtered data provided.
            - Provide a structured summary in bullet points.
            - Keep it precise, professional but discriptive when needed.
            - Use specific values (e.g., revenue, dates) for clarity and impact.
            
            Provide a structured, professional, and engaging summary in a clear narrative format, ensuring concise and insightful takeaways in 4-5 sentences. Also, try to give out the output in bullet points.
            Try to use specific values while giving out recommendations or talking about trends and observations. Show the numbers or values for better understanding of the viewer.
            -also keep the data elements which are fetched from the dataset in bold.
            """

            genai.configure(api_key=api_key)
            model_gen = genai.GenerativeModel(model)
            response = model_gen.generate_content(prompt)
            summary = response.text.strip()
            return summary

        except Exception as e:
            return f"Error processing data or API request: {str(e)}"

    # Secure API Key
    API_KEY = "AIzaSyC27jgTLd_OmJNfwj3uCpvO37n8yM1K-Us" 

    if not API_KEY or API_KEY == "YOUR_GEMINI_API_KEY":
        st.warning("Please configure your Gemini API key.")
        summary = "API key not configured."
    else:
        # Assuming you have a DataFrame named 'df' from your data loading process
        if 'df' not in locals() and 'df' not in globals():
            st.warning("Please load your data first.")
            summary = "No data loaded."
        else:
            with st.spinner("Generating summary..."):
                summary = generate_summary(df, API_KEY)

    st.success(summary)